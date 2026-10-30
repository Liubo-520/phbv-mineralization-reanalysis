"""Run and report strict curve- and study-level validation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.evaluation import grouped_mae_bootstrap_interval, hierarchical_bootstrap_interval, regression_metrics
from src.modeling import nested_evaluate
from src.models import model_catalog
from src.splitting import curve_group_folds, leave_one_study_out_folds


def summarize_predictions(
    predictions: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Summarize each task/model without hiding the weighting level."""
    rows: list[dict] = []
    for group_index, ((task, model), part) in enumerate(
        predictions.groupby(["task", "model"], sort=False)
    ):
        metrics = regression_metrics(part)
        interval = hierarchical_bootstrap_interval(
            part,
            metric="curve_macro_mae",
            iterations=bootstrap_iterations,
            random_state=random_state + group_index,
        )
        study_interval = hierarchical_bootstrap_interval(
            part,
            metric="study_curve_macro_mae",
            iterations=bootstrap_iterations,
            random_state=random_state + 1000 + group_index,
        )
        primary_group = "study" if task == "task_c_leave_one_study_out" else "curve"
        primary_metric = (
            "study_curve_macro_mae" if primary_group == "study" else "curve_macro_mae"
        )
        primary_interval = grouped_mae_bootstrap_interval(
            part,
            group_level=primary_group,
            iterations=bootstrap_iterations,
            random_state=random_state + 2000 + group_index,
        )
        rows.append(
            {
                "task": task,
                "model": model,
                "rows": len(part),
                "studies": part["study_id"].nunique(),
                "curves": part["curve_id"].nunique(),
                **metrics,
                "curve_macro_mae_ci_lower": interval["lower"],
                "curve_macro_mae_ci_upper": interval["upper"],
                "study_curve_macro_mae_ci_lower": study_interval["lower"],
                "study_curve_macro_mae_ci_upper": study_interval["upper"],
                "primary_mae_metric": primary_metric,
                "primary_mae": primary_interval["estimate"],
                "primary_mae_ci_lower": primary_interval["lower"],
                "primary_mae_ci_upper": primary_interval["upper"],
            }
        )
    return pd.DataFrame(rows)


def _grouped_metrics(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, part in predictions.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(dict(zip(group_columns, keys)) | regression_metrics(part) | {"rows": len(part), "curves": part["curve_id"].nunique()})
    return pd.DataFrame(rows)


def run_strict_experiments(
    data_path: Path,
    results_dir: Path,
    model_names: Sequence[str] = ("mean", "ridge", "random_forest", "extra_trees", "xgboost"),
    outer_curve_folds: int = 5,
    inner_curve_folds: int = 4,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
    predictor_set: str = "F-all",
) -> dict[str, pd.DataFrame]:
    """Execute the prespecified Task B and Task C evaluations."""
    data = pd.read_csv(data_path)
    catalog = model_catalog(predictor_set)
    unknown = set(model_names).difference(catalog)
    if unknown:
        raise ValueError(f"Unknown requested models: {sorted(unknown)}")

    task_folds = {
        "task_b_curve_grouped_nested_cv": {
            "folds": curve_group_folds(
                data, n_splits=outer_curve_folds, group_column="curve_family_id"
            ),
            "inner_group_column": "curve_family_id",
            "selection_metric": "curve_macro_mae",
            "weighting": "curve",
        },
        "task_c_leave_one_study_out": {
            "folds": leave_one_study_out_folds(data),
            "inner_group_column": "study_id",
            "selection_metric": "study_curve_macro_mae",
            "weighting": "study",
        },
    }
    prediction_parts: list[pd.DataFrame] = []
    tuning_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for task, design in task_folds.items():
        for model_name in model_names:
            prediction, tuning, audit = nested_evaluate(
                data,
                spec=catalog[model_name],
                outer_folds=design["folds"],
                task=task,
                inner_splits=inner_curve_folds,
                inner_group_column=design["inner_group_column"],
                selection_metric=design["selection_metric"],
                weighting=design["weighting"],
                random_state=random_state,
            )
            prediction_parts.append(prediction)
            tuning_parts.append(tuning)
            audit_parts.append(audit)
            print(f"strict: {task} / {model_name} complete", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    fold_audit = pd.concat(audit_parts, ignore_index=True)
    summary = summarize_predictions(predictions, bootstrap_iterations, random_state)
    fold_metrics = _grouped_metrics(predictions, ["task", "model", "outer_fold"])
    study_metrics = _grouped_metrics(predictions, ["task", "model", "study_id"])

    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "tuning").mkdir(parents=True, exist_ok=True)
    (results_dir / "audit").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / "predictions/strict_oof_predictions.csv", index=False)
    tuning.to_csv(results_dir / "tuning/strict_nested_tuning.csv", index=False)
    fold_audit.to_csv(results_dir / "audit/strict_fold_audit.csv", index=False)
    summary.to_csv(results_dir / "metrics/strict_model_performance.csv", index=False)
    fold_metrics.to_csv(results_dir / "metrics/strict_outer_fold_metrics.csv", index=False)
    study_metrics.to_csv(results_dir / "metrics/strict_per_study_metrics.csv", index=False)
    metadata = {
        "primary_dataset": str(data_path),
        "model_names": list(model_names),
        "outer_curve_folds": outer_curve_folds,
        "inner_curve_folds": inner_curve_folds,
        "bootstrap_iterations": bootstrap_iterations,
        "random_state": random_state,
        "predictor_set": predictor_set,
        "task_b_selection_metric": "curve_macro_mae",
        "task_c_selection_metric": "study_curve_macro_mae",
        "task_b_sample_weighting": "equal total weight per curve",
        "task_c_sample_weighting": "equal total weight per study, then per curve within study",
        "task_b_inner_group": "curve_family_id (identical recorded feature profiles kept together)",
        "task_c_inner_group": "study_id",
    }
    (results_dir / "audit/strict_experiment_design.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "predictions": predictions,
        "tuning": tuning,
        "fold_audit": fold_audit,
        "summary": summary,
        "fold_metrics": fold_metrics,
        "study_metrics": study_metrics,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_strict_experiments(
        root / "data/processed/D3_endpoint_audited_mineralization.csv",
        root / "results",
    )
