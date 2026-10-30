"""Curve-grouped split-conformal uncertainty for frozen RF predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.models import make_estimator, model_catalog, sample_weights
from src.preprocessing import build_preprocessor
from src.sensitivity import FROZEN_RF_PARAMETERS
from src.splitting import leave_one_study_out_folds


def finite_sample_quantile(scores: Sequence[float], coverage: float = 0.90) -> float:
    """Return the conformal higher quantile with finite-calibration correction."""
    values = np.sort(np.asarray(scores, dtype=float))
    if len(values) == 0:
        raise ValueError("At least one calibration score is required")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(values[rank - 1])


def curve_maximum_scores(calibration: pd.DataFrame) -> pd.Series:
    errors = (calibration["y_true"] - calibration["y_pred"]).abs()
    return errors.groupby(calibration["curve_id"], sort=False).max()


def interval_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:
    covered = predictions["y_true"].between(predictions["lower"], predictions["upper"])
    curve_covered = covered.groupby(predictions["curve_id"], sort=False).all()
    diagnostics = {
        "point_coverage": float(covered.mean()),
        "curve_simultaneous_coverage": float(curve_covered.mean()),
        "mean_interval_width": float((predictions["upper"] - predictions["lower"]).mean()),
    }
    if "study_id" in predictions:
        curve_study = pd.DataFrame(
            {
                "study_id": predictions["study_id"].to_numpy(),
                "curve_id": predictions["curve_id"].to_numpy(),
                "covered": covered.to_numpy(),
            }
        ).groupby(["study_id", "curve_id"], sort=False)["covered"].all()
        diagnostics["study_curve_simultaneous_coverage"] = float(
            curve_study.groupby(level="study_id", sort=False).mean().mean()
        )
    return diagnostics


def _curve_mapping_folds(data: pd.DataFrame, reference: pd.DataFrame):
    mapping = reference.drop_duplicates("curve_id").set_index("curve_id")["outer_fold"]
    assigned = data["curve_id"].map(mapping)
    if assigned.isna().any():
        raise ValueError("A D1 curve has no reference outer fold")
    return [
        (
            np.flatnonzero(assigned.to_numpy() != fold),
            np.flatnonzero(assigned.to_numpy() == fold),
        )
        for fold in sorted(assigned.unique())
    ]


def grouped_calibration_split(
    data: pd.DataFrame,
    *,
    group_column: str,
    calibration_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete groups into proper-training and calibration partitions."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=calibration_fraction,
        random_state=random_state,
    )
    proper_pos, calibration_pos = next(
        splitter.split(data, groups=data[group_column])
    )
    return data.iloc[proper_pos], data.iloc[calibration_pos]


def run_grouped_conformal(
    root: Path,
    coverage: float = 0.90,
    calibration_fraction: float = 0.20,
    random_state: int = 42,
    predictor_set: str = "F-all",
) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    reference_predictions = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    reference = reference_predictions.loc[
        reference_predictions["task"].eq("task_b_curve_grouped_nested_cv")
        & reference_predictions["model"].eq("random_forest"),
        ["curve_id", "outer_fold"],
    ]
    tasks = {
        "task_b_curve_grouped_conformal": {
            "folds": _curve_mapping_folds(data, reference),
            "calibration_group": "curve_family_id",
            "weighting": "curve",
        },
        "task_c_loso_conformal": {
            "folds": leave_one_study_out_folds(data),
            "calibration_group": "study_id",
            "weighting": "study",
        },
    }
    spec = model_catalog(predictor_set)["random_forest"]
    features = list(spec.numeric_features + spec.categorical_features)
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for task, design in tasks.items():
        folds = design["folds"]
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(folds):
            outer_train = data.iloc[outer_train_idx]
            outer_test = data.iloc[outer_test_idx]
            proper, calibration = grouped_calibration_split(
                outer_train,
                group_column=design["calibration_group"],
                calibration_fraction=calibration_fraction,
                random_state=random_state + outer_fold,
            )
            proper_curves = set(proper["curve_id"])
            calibration_curves = set(calibration["curve_id"])
            test_curves = set(outer_test["curve_id"])
            if proper_curves & calibration_curves or proper_curves & test_curves or calibration_curves & test_curves:
                raise ValueError("Curve leakage detected in conformal split")

            preprocessor = build_preprocessor(spec.numeric_features, spec.categorical_features)
            x_proper = preprocessor.fit_transform(proper[features])
            x_calibration = preprocessor.transform(calibration[features])
            x_test = preprocessor.transform(outer_test[features])
            estimator = make_estimator(
                "random_forest",
                FROZEN_RF_PARAMETERS,
                random_state=random_state + outer_fold,
            )
            estimator.fit(
                x_proper,
                proper["biodegradation_percentage"].to_numpy(dtype=float),
                sample_weight=sample_weights(proper, design["weighting"]),
            )
            calibration_frame = pd.DataFrame(
                {
                    "curve_id": calibration["curve_id"].to_numpy(),
                    "y_true": calibration["biodegradation_percentage"].to_numpy(dtype=float),
                    "y_pred": estimator.predict(x_calibration),
                }
            )
            scores = curve_maximum_scores(calibration_frame)
            radius = finite_sample_quantile(scores, coverage=coverage)
            center = np.clip(estimator.predict(x_test), 0.0, 100.0)
            fold_predictions = pd.DataFrame(
                {
                    "task": task,
                    "outer_fold": outer_fold,
                    "row_index": np.asarray(outer_test_idx, dtype=int),
                    "study_id": outer_test["study_id"].to_numpy(),
                    "curve_id": outer_test["curve_id"].to_numpy(),
                    "y_true": outer_test["biodegradation_percentage"].to_numpy(dtype=float),
                    "y_pred": center,
                    "lower": np.clip(center - radius, 0.0, 100.0),
                    "upper": np.clip(center + radius, 0.0, 100.0),
                    "conformal_radius": radius,
                    "calibration_curves": len(scores),
                    "nominal_curve_coverage": coverage,
                }
            )
            prediction_parts.append(fold_predictions)
            audit_rows.append(
                {
                    "task": task,
                    "outer_fold": outer_fold,
                    "proper_train_rows": len(proper),
                    "proper_train_curves": len(proper_curves),
                    "calibration_rows": len(calibration),
                    "calibration_curves": len(calibration_curves),
                    "test_rows": len(outer_test),
                    "test_curves": len(test_curves),
                    "proper_calibration_curve_overlap": len(proper_curves & calibration_curves),
                    "proper_test_curve_overlap": len(proper_curves & test_curves),
                    "calibration_test_curve_overlap": len(calibration_curves & test_curves),
                    "calibration_group": design["calibration_group"],
                    "weighting": design["weighting"],
                    "proper_calibration_study_overlap": len(
                        set(proper["study_id"]) & set(calibration["study_id"])
                    ),
                    "conformal_radius": radius,
                }
            )
            print(f"conformal: {task} fold {outer_fold + 1}/{len(folds)} complete", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_audit = pd.DataFrame(audit_rows)
    fold_rows = []
    for (task, outer_fold), part in predictions.groupby(["task", "outer_fold"], sort=False):
        fold_rows.append(
            {"task": task, "outer_fold": outer_fold, "rows": len(part), "curves": part["curve_id"].nunique()}
            | interval_diagnostics(part)
        )
    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for task, part in predictions.groupby("task", sort=False):
        summary_rows.append(
            {
                "task": task,
                "nominal_curve_coverage": coverage,
                "rows": len(part),
                "curves": part["curve_id"].nunique(),
                "studies": part["study_id"].nunique(),
                "median_conformal_radius": float(part.groupby("outer_fold")["conformal_radius"].first().median()),
            }
            | interval_diagnostics(part)
        )
    summary = pd.DataFrame(summary_rows)

    (root / "results/predictions").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    (root / "results/audit").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(root / "results/predictions/grouped_conformal_predictions.csv", index=False)
    fold_metrics.to_csv(root / "results/metrics/grouped_conformal_fold_metrics.csv", index=False)
    summary.to_csv(root / "results/metrics/grouped_conformal_summary.csv", index=False)
    fold_audit.to_csv(root / "results/audit/grouped_conformal_fold_audit.csv", index=False)
    return {"predictions": predictions, "fold_metrics": fold_metrics, "summary": summary, "audit": fold_audit}


if __name__ == "__main__":
    run_grouped_conformal(Path(__file__).resolve().parents[1])
