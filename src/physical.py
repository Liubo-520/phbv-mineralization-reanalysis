"""Target-blind physical diagnostics and prediction post-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.evaluation import regression_metrics


def _group_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in ("task", "model", "curve_id") if column in data.columns]


def apply_physical_postprocessing(
    predictions: pd.DataFrame,
    prediction_column: str = "y_pred",
) -> pd.DataFrame:
    """Add bounded and bounded-monotone predictions without reading targets."""
    required = {"curve_id", "biodegradation_time_days", prediction_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing physical-processing columns: {sorted(missing)}")
    output = predictions.copy()
    output["y_pred_clipped"] = output[prediction_column].clip(0.0, 100.0)
    output["y_pred_monotone"] = np.nan
    for _, part in output.groupby(_group_columns(output), sort=False):
        ordered = part.sort_values("biodegradation_time_days")
        isotonic = IsotonicRegression(increasing=True, y_min=0.0, y_max=100.0, out_of_bounds="clip")
        corrected = isotonic.fit_transform(
            ordered["biodegradation_time_days"].to_numpy(dtype=float),
            ordered["y_pred_clipped"].to_numpy(dtype=float),
        )
        output.loc[ordered.index, "y_pred_monotone"] = corrected
    return output


def physical_diagnostics(
    predictions: pd.DataFrame,
    prediction_column: str = "y_pred",
) -> dict[str, int]:
    """Count bound violations and time-adjacent decreases within curves."""
    values = predictions[prediction_column]
    rows_outside = int((values.lt(0.0) | values.gt(100.0)).sum())
    comparisons = 0
    violations = 0
    curves_with_violation = 0
    for _, part in predictions.groupby(_group_columns(predictions), sort=False):
        ordered = part.sort_values("biodegradation_time_days")
        decreases = ordered[prediction_column].diff().dropna().lt(-1e-12)
        comparisons += len(decreases)
        violations += int(decreases.sum())
        curves_with_violation += int(decreases.any())
    return {
        "rows_outside_bounds": rows_outside,
        "adjacent_comparisons": comparisons,
        "monotonicity_violations": violations,
        "curves_with_violation": curves_with_violation,
    }


def summarize_physical_variants(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report predictive and physical metrics for every declared variant."""
    variants = [
        ("raw", "y_pred"),
        ("bounded", "y_pred_clipped"),
        ("bounded_monotone", "y_pred_monotone"),
    ]
    rows: list[dict] = []
    for (task, model), part in predictions.groupby(["task", "model"], sort=False):
        for variant, column in variants:
            metric_input = part.copy()
            metric_input["y_pred"] = metric_input[column]
            rows.append(
                {"task": task, "model": model, "variant": variant}
                | regression_metrics(metric_input)
                | physical_diagnostics(part, prediction_column=column)
            )
    return pd.DataFrame(rows)


def run_physical_analysis(data_path: Path, predictions_path: Path, results_dir: Path) -> dict[str, pd.DataFrame]:
    """Attach times to OOF predictions, post-process, summarize, and save."""
    data = pd.read_csv(data_path).reset_index(names="row_index")
    predictions = pd.read_csv(predictions_path)
    predictions = predictions.merge(
        data[["row_index", "biodegradation_time_days"]],
        on="row_index",
        how="left",
        validate="many_to_one",
    )
    if predictions["biodegradation_time_days"].isna().any():
        raise ValueError("At least one prediction could not be linked to time")
    processed = apply_physical_postprocessing(predictions)
    summary = summarize_physical_variants(processed)
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    processed.to_csv(results_dir / "predictions/strict_oof_predictions_physical.csv", index=False)
    summary.to_csv(results_dir / "metrics/physical_postprocessing_comparison.csv", index=False)
    return {"predictions": processed, "summary": summary}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_physical_analysis(
        root / "data/processed/D3_endpoint_audited_mineralization.csv",
        root / "results/predictions/strict_oof_predictions.csv",
        root / "results",
    )
