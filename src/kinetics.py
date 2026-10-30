"""Bounded first-order mineralization kinetics for observed and predicted curves."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import spearmanr


def _first_order(time: np.ndarray, asymptote: float, rate: float) -> np.ndarray:
    return asymptote * (1.0 - np.exp(-rate * time))


def fit_first_order_kinetics(time, response) -> dict[str, float | str | int]:
    frame = pd.DataFrame({"time": np.asarray(time, dtype=float), "response": np.asarray(response, dtype=float)})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().groupby("time", as_index=False)["response"].mean()
    if len(frame) < 4:
        return {
            "status": "insufficient_unique_times",
            "unique_times": len(frame),
            "asymptote_percent": np.nan,
            "rate_per_day": np.nan,
            "half_time_days": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }
    x = frame["time"].to_numpy()
    y = frame["response"].to_numpy()
    initial_asymptote = min(99.0, max(1.0, float(y.max()) * 1.05))
    positive_times = x[x > 0]
    initial_rate = np.log(2.0) / np.median(positive_times) if len(positive_times) else 0.01
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            parameters, _ = curve_fit(
                _first_order,
                x,
                y,
                p0=[initial_asymptote, initial_rate],
                bounds=([1e-6, 1e-8], [100.0, 10.0]),
                maxfev=20000,
            )
        prediction = _first_order(x, *parameters)
        residual = y - prediction
        denominator = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else np.nan
        asymptote, rate = map(float, parameters)
        return {
            "status": "ok",
            "unique_times": len(frame),
            "asymptote_percent": asymptote,
            "rate_per_day": rate,
            "half_time_days": float(np.log(2.0) / rate),
            "r2": float(r2),
            "rmse": float(np.sqrt(np.mean(residual**2))),
        }
    except (RuntimeError, ValueError, FloatingPointError):
        return {
            "status": "fit_failed",
            "unique_times": len(frame),
            "asymptote_percent": np.nan,
            "rate_per_day": np.nan,
            "half_time_days": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
        }


def _fit_curve_table(data: pd.DataFrame, value_column: str, source: str) -> pd.DataFrame:
    rows = []
    for curve_id, part in data.groupby("curve_id", sort=False):
        fit = fit_first_order_kinetics(part["biodegradation_time_days"], part[value_column])
        rows.append(
            {
                "source": source,
                "study_id": int(part["study_id"].iloc[0]),
                "curve_id": curve_id,
                "observations": len(part),
            }
            | fit
        )
    return pd.DataFrame(rows)


def run_kinetic_analysis(root: Path) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    physical = pd.read_csv(root / "results/predictions/strict_oof_predictions_physical.csv")
    rf = physical[physical["model"].eq("random_forest")]
    fit_parts = [_fit_curve_table(data, "biodegradation_percentage", "observed")]
    source_map = {
        "task_b_curve_grouped_nested_cv": "rf_curve_grouped_cv",
        "task_c_leave_one_study_out": "rf_loso",
    }
    for task, source in source_map.items():
        fit_parts.append(_fit_curve_table(rf[rf["task"].eq(task)], "y_pred_monotone", source))
    fits = pd.concat(fit_parts, ignore_index=True)

    summary_rows = []
    for source, part in fits.groupby("source", sort=False):
        valid = part[part["status"].eq("ok")]
        summary_rows.append(
            {
                "source": source,
                "curves_total": len(part),
                "curves_fit": len(valid),
                "median_fit_r2": valid["r2"].median(),
                "median_fit_rmse": valid["rmse"].median(),
                "asymptote_boundary_hits": int(valid["asymptote_percent"].ge(99.9).sum()),
                "median_rate_per_day": valid["rate_per_day"].median(),
                "median_half_time_days": valid["half_time_days"].median(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    comparisons = []
    observed = fits[(fits["source"].eq("observed")) & fits["status"].eq("ok")]
    for source in ("rf_curve_grouped_cv", "rf_loso"):
        predicted = fits[(fits["source"].eq(source)) & fits["status"].eq("ok")]
        merged = observed.merge(predicted, on=["study_id", "curve_id"], suffixes=("_observed", "_predicted"))
        positive = merged[(merged["rate_per_day_observed"] > 0) & (merged["rate_per_day_predicted"] > 0)]
        rate_rho, rate_p = spearmanr(np.log(positive["rate_per_day_observed"]), np.log(positive["rate_per_day_predicted"]))
        asymptote_rho, asymptote_p = spearmanr(merged["asymptote_percent_observed"], merged["asymptote_percent_predicted"])
        comparisons.append(
            {
                "source": source,
                "paired_curves": len(merged),
                "log_rate_spearman_rho": rate_rho,
                "log_rate_p_value": rate_p,
                "rate_median_absolute_log_error": float(np.median(np.abs(np.log(positive["rate_per_day_observed"]) - np.log(positive["rate_per_day_predicted"])))),
                "asymptote_spearman_rho": asymptote_rho,
                "asymptote_p_value": asymptote_p,
                "asymptote_mae": float(np.mean(np.abs(merged["asymptote_percent_observed"] - merged["asymptote_percent_predicted"]))),
            }
        )
    comparison = pd.DataFrame(comparisons)

    (root / "results/kinetics").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    fits.to_csv(root / "results/kinetics/first_order_curve_fits.csv", index=False)
    summary.to_csv(root / "results/metrics/kinetic_fit_summary.csv", index=False)
    comparison.to_csv(root / "results/metrics/kinetic_parameter_fidelity.csv", index=False)
    return {"fits": fits, "summary": summary, "comparison": comparison}


if __name__ == "__main__":
    run_kinetic_analysis(Path(__file__).resolve().parents[1])
