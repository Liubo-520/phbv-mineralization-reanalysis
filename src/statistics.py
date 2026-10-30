"""Cluster-aligned statistical comparisons for model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * sorted_values
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))


def paired_curve_comparisons(
    predictions: pd.DataFrame,
    reference_model: str,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 20260726,
) -> pd.DataFrame:
    """Compare each model with a reference using curve-level absolute error.

    Differences are candidate minus reference, so positive values favor the
    reference. Bootstrap intervals resample curves, the independent unit.
    """
    required = {"task", "model", "curve_id", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    curve_losses = (
        predictions.assign(abs_error=lambda frame: np.abs(frame.y_true - frame.y_pred))
        .groupby(["task", "model", "curve_id"], as_index=False)
        .agg(curve_mae=("abs_error", "mean"))
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for task, task_frame in curve_losses.groupby("task", sort=True):
        reference = task_frame[task_frame.model == reference_model][
            ["curve_id", "curve_mae"]
        ].rename(columns={"curve_mae": "reference_mae"})
        if reference.empty:
            raise ValueError(f"Reference model {reference_model!r} absent from {task!r}")
        for candidate in sorted(set(task_frame.model) - {reference_model}):
            candidate_loss = task_frame[task_frame.model == candidate][
                ["curve_id", "curve_mae"]
            ].rename(columns={"curve_mae": "candidate_mae"})
            paired = reference.merge(candidate_loss, on="curve_id", how="inner")
            differences = (paired.candidate_mae - paired.reference_mae).to_numpy()
            if not len(differences):
                continue
            if np.allclose(differences, 0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(differences, alternative="two-sided", method="auto").pvalue
                )
            samples = rng.choice(
                differences,
                size=(bootstrap_repeats, len(differences)),
                replace=True,
            )
            bootstrap_medians = np.median(samples, axis=1)
            rows.append(
                {
                    "task": task,
                    "reference_model": reference_model,
                    "candidate_model": candidate,
                    "paired_curves": len(differences),
                    "mean_mae_difference": float(np.mean(differences)),
                    "median_mae_difference": float(np.median(differences)),
                    "median_difference_ci_lower": float(
                        np.quantile(bootstrap_medians, 0.025)
                    ),
                    "median_difference_ci_upper": float(
                        np.quantile(bootstrap_medians, 0.975)
                    ),
                    "better_curve_fraction": float(np.mean(differences < 0)),
                    "rank_biserial": _rank_biserial(differences),
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_value_holm"] = np.nan
    for task, indices in result.groupby("task").groups.items():
        del task
        result.loc[indices, "p_value_holm"] = holm_adjust(
            result.loc[indices, "p_value"].to_numpy()
        )
    return result


def paired_study_comparisons(
    predictions: pd.DataFrame,
    reference_model: str,
    *,
    bootstrap_repeats: int = 10_000,
    seed: int = 20260726,
) -> pd.DataFrame:
    """Compare models using one curve-balanced loss per independent study.

    Curves are first assigned equal weight within their study. Wilcoxon tests
    and bootstrap intervals then operate on study-level paired differences.
    This is the primary comparison for study-transport claims.
    """
    required = {"task", "model", "study_id", "curve_id", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    curve_losses = (
        predictions.assign(abs_error=lambda frame: np.abs(frame.y_true - frame.y_pred))
        .groupby(["task", "model", "study_id", "curve_id"], as_index=False)
        .agg(curve_mae=("abs_error", "mean"))
    )
    study_losses = (
        curve_losses.groupby(["task", "model", "study_id"], as_index=False)
        .agg(study_curve_macro_mae=("curve_mae", "mean"), curves=("curve_id", "nunique"))
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for task, task_frame in study_losses.groupby("task", sort=True):
        reference = task_frame[task_frame.model == reference_model][
            ["study_id", "study_curve_macro_mae", "curves"]
        ].rename(
            columns={
                "study_curve_macro_mae": "reference_mae",
                "curves": "reference_curves",
            }
        )
        if reference.empty:
            raise ValueError(f"Reference model {reference_model!r} absent from {task!r}")
        for candidate in sorted(set(task_frame.model) - {reference_model}):
            candidate_loss = task_frame[task_frame.model == candidate][
                ["study_id", "study_curve_macro_mae", "curves"]
            ].rename(
                columns={
                    "study_curve_macro_mae": "candidate_mae",
                    "curves": "candidate_curves",
                }
            )
            paired = reference.merge(candidate_loss, on="study_id", how="inner")
            differences = (paired.candidate_mae - paired.reference_mae).to_numpy()
            if not len(differences):
                continue
            if np.allclose(differences, 0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(differences, alternative="two-sided", method="auto").pvalue
                )
            samples = rng.choice(
                differences, size=(bootstrap_repeats, len(differences)), replace=True
            )
            bootstrap_medians = np.median(samples, axis=1)
            rows.append(
                {
                    "task": task,
                    "reference_model": reference_model,
                    "candidate_model": candidate,
                    "paired_studies": len(differences),
                    "paired_curves": int(paired["reference_curves"].sum()),
                    "mean_mae_difference": float(np.mean(differences)),
                    "median_mae_difference": float(np.median(differences)),
                    "median_difference_ci_lower": float(
                        np.quantile(bootstrap_medians, 0.025)
                    ),
                    "median_difference_ci_upper": float(
                        np.quantile(bootstrap_medians, 0.975)
                    ),
                    "better_study_fraction": float(np.mean(differences < 0)),
                    "rank_biserial": _rank_biserial(differences),
                    "p_value": p_value,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_value_holm"] = np.nan
    for task, indices in result.groupby("task").groups.items():
        del task
        result.loc[indices, "p_value_holm"] = holm_adjust(
            result.loc[indices, "p_value"].to_numpy()
        )
    return result
