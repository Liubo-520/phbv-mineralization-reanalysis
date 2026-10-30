"""Leakage-resistant nested evaluation with fold-local preprocessing."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.models import ModelSpec, make_estimator, sample_weights
from src.preprocessing import build_preprocessor
from src.splitting import IndexPair, curve_group_folds


def _prepared_split(train: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec):
    features = list(spec.numeric_features + spec.categorical_features)
    preprocessor = build_preprocessor(spec.numeric_features, spec.categorical_features)
    x_train = preprocessor.fit_transform(train[features])
    x_test = preprocessor.transform(test[features])
    return x_train, x_test


def _prediction_frame(data: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_id": data["study_id"].to_numpy(),
            "curve_id": data["curve_id"].to_numpy(),
            "y_true": data["biodegradation_percentage"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )


def nested_evaluate(
    data: pd.DataFrame,
    spec: ModelSpec,
    outer_folds: Iterable[IndexPair],
    task: str,
    inner_splits: int = 4,
    inner_group_column: str = "curve_id",
    selection_metric: str = "curve_macro_mae",
    weighting: str = "curve",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune within each outer training set and return auditable OOF predictions."""
    predictions: list[pd.DataFrame] = []
    tuning_rows: list[dict] = []
    audit_rows: list[dict] = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_folds):
        outer_train = data.iloc[outer_train_idx].copy()
        outer_test = data.iloc[outer_test_idx].copy()
        train_curves = set(outer_train["curve_id"])
        test_curves = set(outer_test["curve_id"])
        train_studies = set(outer_train["study_id"])
        test_studies = set(outer_test["study_id"])
        train_families = set(outer_train.get("curve_family_id", outer_train["curve_id"]))
        test_families = set(outer_test.get("curve_family_id", outer_test["curve_id"]))

        if inner_group_column not in outer_train:
            raise ValueError(f"Missing inner grouping column: {inner_group_column}")
        n_inner = min(inner_splits, outer_train[inner_group_column].nunique())
        if n_inner < 2:
            raise ValueError("At least two training groups are required for inner CV")
        inner_folds = curve_group_folds(
            outer_train, n_splits=n_inner, group_column=inner_group_column
        )
        prepared = []
        inner_group_overlap_count = 0
        for inner_train_idx, inner_valid_idx in inner_folds:
            inner_train = outer_train.iloc[inner_train_idx]
            inner_valid = outer_train.iloc[inner_valid_idx]
            inner_group_overlap_count += len(
                set(inner_train[inner_group_column]) & set(inner_valid[inner_group_column])
            )
            x_train, x_valid = _prepared_split(inner_train, inner_valid, spec)
            prepared.append((inner_train, inner_valid, x_train, x_valid))

        candidate_scores: list[float] = []
        for candidate_index, parameters in enumerate(spec.candidates):
            candidate_predictions: list[pd.DataFrame] = []
            for inner_fold, (inner_train, inner_valid, x_train, x_valid) in enumerate(prepared):
                estimator = make_estimator(
                    spec.name,
                    parameters,
                    random_state=random_state + outer_fold * 1000 + candidate_index * 10 + inner_fold,
                )
                estimator.fit(
                    x_train,
                    inner_train["biodegradation_percentage"].to_numpy(dtype=float),
                    sample_weight=sample_weights(inner_train, weighting),
                )
                candidate_predictions.append(_prediction_frame(inner_valid, estimator.predict(x_valid)))
            inner_predictions = pd.concat(candidate_predictions, ignore_index=True)
            score = regression_metrics(inner_predictions)[selection_metric]
            candidate_scores.append(score)
            tuning_rows.append(
                {
                    "task": task,
                    "model": spec.name,
                    "outer_fold": outer_fold,
                    "candidate_index": candidate_index,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "inner_selection_score": score,
                    "selection_metric": selection_metric,
                    "inner_group_column": inner_group_column,
                    "weighting": weighting,
                    "selected": False,
                }
            )

        best_index = int(np.nanargmin(candidate_scores))
        tuning_rows[-len(spec.candidates) + best_index]["selected"] = True
        best_parameters = spec.candidates[best_index]

        x_train, x_test = _prepared_split(outer_train, outer_test, spec)
        estimator = make_estimator(
            spec.name,
            best_parameters,
            random_state=random_state + outer_fold,
        )
        estimator.fit(
            x_train,
            outer_train["biodegradation_percentage"].to_numpy(dtype=float),
            sample_weight=sample_weights(outer_train, weighting),
        )
        fold_predictions = _prediction_frame(outer_test, estimator.predict(x_test))
        fold_predictions.insert(0, "row_index", np.asarray(outer_test_idx, dtype=int))
        fold_predictions.insert(0, "outer_fold", outer_fold)
        fold_predictions.insert(0, "model", spec.name)
        fold_predictions.insert(0, "task", task)
        fold_predictions["best_parameters"] = json.dumps(best_parameters, sort_keys=True)
        predictions.append(fold_predictions)

        audit_rows.append(
            {
                "task": task,
                "model": spec.name,
                "outer_fold": outer_fold,
                "train_rows": len(outer_train),
                "test_rows": len(outer_test),
                "train_curves": len(train_curves),
                "test_curves": len(test_curves),
                "curve_overlap_count": len(train_curves & test_curves),
                "curve_family_overlap_count": len(train_families & test_families),
                "train_studies": len(train_studies),
                "test_studies": len(test_studies),
                "study_overlap_count": len(train_studies & test_studies),
                "inner_group_column": inner_group_column,
                "inner_group_overlap_count": inner_group_overlap_count,
                "selection_metric": selection_metric,
                "weighting": weighting,
            }
        )

    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(audit_rows),
    )


def fixed_evaluate(
    data: pd.DataFrame,
    spec: ModelSpec,
    parameters: dict,
    outer_folds: Iterable[IndexPair],
    task: str,
    weighting: str = "curve",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a frozen candidate once per outer fold, without retuning."""
    predictions: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    encoded_parameters = json.dumps(parameters, sort_keys=True)
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_folds):
        outer_train = data.iloc[outer_train_idx].copy()
        outer_test = data.iloc[outer_test_idx].copy()
        x_train, x_test = _prepared_split(outer_train, outer_test, spec)
        estimator = make_estimator(spec.name, parameters, random_state=random_state + outer_fold)
        estimator.fit(
            x_train,
            outer_train["biodegradation_percentage"].to_numpy(dtype=float),
            sample_weight=sample_weights(outer_train, weighting),
        )
        fold_predictions = _prediction_frame(outer_test, estimator.predict(x_test))
        fold_predictions.insert(0, "row_index", np.asarray(outer_test_idx, dtype=int))
        fold_predictions.insert(0, "outer_fold", outer_fold)
        fold_predictions.insert(0, "model", spec.name)
        fold_predictions.insert(0, "task", task)
        fold_predictions["best_parameters"] = encoded_parameters
        predictions.append(fold_predictions)

        train_curves = set(outer_train["curve_id"])
        test_curves = set(outer_test["curve_id"])
        train_studies = set(outer_train["study_id"])
        test_studies = set(outer_test["study_id"])
        train_families = set(outer_train.get("curve_family_id", outer_train["curve_id"]))
        test_families = set(outer_test.get("curve_family_id", outer_test["curve_id"]))
        audit_rows.append(
            {
                "task": task,
                "model": spec.name,
                "outer_fold": outer_fold,
                "train_rows": len(outer_train),
                "test_rows": len(outer_test),
                "train_curves": len(train_curves),
                "test_curves": len(test_curves),
                "curve_overlap_count": len(train_curves & test_curves),
                "curve_family_overlap_count": len(train_families & test_families),
                "train_studies": len(train_studies),
                "test_studies": len(test_studies),
                "study_overlap_count": len(train_studies & test_studies),
                "weighting": weighting,
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit_rows)
