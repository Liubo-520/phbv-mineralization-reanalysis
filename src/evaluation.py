"""Metrics that expose point-, curve-, and study-weighting choices."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return np.nan
    return float(r2_score(y_true, y_pred))


def _group_metric_mean(data: pd.DataFrame, group: str, metric: str) -> float:
    values: list[float] = []
    for _, part in data.groupby(group, sort=False):
        if metric == "mae":
            value = mean_absolute_error(part["y_true"], part["y_pred"])
        elif metric == "rmse":
            value = np.sqrt(mean_squared_error(part["y_true"], part["y_pred"]))
        elif metric == "r2":
            value = _safe_r2(part["y_true"], part["y_pred"])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        values.append(float(value))
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute complementary pooled and hierarchy-balanced regression metrics."""
    required = {"study_id", "curve_id", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("Predictions must not be empty")

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    metrics = {
        "point_mae": float(mean_absolute_error(y_true, y_pred)),
        "point_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "point_r2": _safe_r2(y_true, y_pred),
    }
    for level, column in (("curve_macro", "curve_id"), ("study_macro", "study_id")):
        for metric in ("mae", "rmse", "r2"):
            metrics[f"{level}_{metric}"] = _group_metric_mean(predictions, column, metric)
    curve_mae = (
        predictions.assign(_absolute_error=(y_true - y_pred).abs())
        .groupby(["study_id", "curve_id"], sort=False)["_absolute_error"]
        .mean()
    )
    metrics["study_curve_macro_mae"] = float(
        curve_mae.groupby(level="study_id", sort=False).mean().mean()
    )
    return metrics


def hierarchical_bootstrap_interval(
    predictions: pd.DataFrame,
    metric: str = "curve_macro_mae",
    iterations: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap studies, then complete curves within each sampled study."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    estimate = regression_metrics(predictions)[metric]
    studies = predictions["study_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    draws: list[float] = []
    curve_mae_by_study: dict[object, np.ndarray] | None = None
    study_curve_mae: np.ndarray | None = None
    if metric == "curve_macro_mae":
        curve_errors = predictions.assign(
            _absolute_error=(predictions["y_true"] - predictions["y_pred"]).abs()
        ).groupby(["study_id", "curve_id"], sort=False, as_index=False)["_absolute_error"].mean()
        curve_mae_by_study = {
            study: curve_errors.loc[curve_errors["study_id"].eq(study), "_absolute_error"].to_numpy()
            for study in studies
        }
    elif metric == "study_curve_macro_mae":
        curve_errors = predictions.assign(
            _absolute_error=(predictions["y_true"] - predictions["y_pred"]).abs()
        ).groupby(["study_id", "curve_id"], sort=False, as_index=False)["_absolute_error"].mean()
        study_curve_mae = (
            curve_errors.groupby("study_id", sort=False)["_absolute_error"].mean().to_numpy()
        )
    for _ in range(iterations):
        if study_curve_mae is not None:
            draws.append(float(rng.choice(study_curve_mae, size=len(study_curve_mae), replace=True).mean()))
            continue
        sampled_studies = rng.choice(studies, size=len(studies), replace=True)
        if curve_mae_by_study is not None:
            sampled_errors = []
            for study in sampled_studies:
                errors = curve_mae_by_study[study]
                sampled_errors.append(rng.choice(errors, size=len(errors), replace=True))
            draws.append(float(np.concatenate(sampled_errors).mean()))
            continue
        pieces: list[pd.DataFrame] = []
        for study_draw, study in enumerate(sampled_studies):
            study_data = predictions.loc[predictions["study_id"].eq(study)]
            curves = study_data["curve_id"].drop_duplicates().to_numpy()
            for curve_draw, curve in enumerate(rng.choice(curves, size=len(curves), replace=True)):
                piece = study_data.loc[study_data["curve_id"].eq(curve)].copy()
                piece["study_id"] = study_draw
                piece["curve_id"] = f"{study_draw}:{curve_draw}"
                pieces.append(piece)
        value = regression_metrics(pd.concat(pieces, ignore_index=True))[metric]
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        raise ValueError(f"No finite bootstrap values for {metric}")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return {"estimate": float(estimate), "lower": float(lower), "upper": float(upper)}


def grouped_mae_bootstrap_interval(
    predictions: pd.DataFrame,
    *,
    group_level: str,
    iterations: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Bootstrap the declared independent unit for a hierarchy-aligned MAE."""
    if group_level == "curve":
        values = (
            predictions.assign(_ae=(predictions.y_true - predictions.y_pred).abs())
            .groupby("curve_id", sort=False)["_ae"]
            .mean()
            .to_numpy()
        )
    elif group_level == "study":
        curve_values = (
            predictions.assign(_ae=(predictions.y_true - predictions.y_pred).abs())
            .groupby(["study_id", "curve_id"], sort=False)["_ae"]
            .mean()
        )
        values = curve_values.groupby(level="study_id", sort=False).mean().to_numpy()
    else:
        raise ValueError("group_level must be 'curve' or 'study'")
    if iterations < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(random_state)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return {"estimate": float(values.mean()), "lower": float(lower), "upper": float(upper)}
