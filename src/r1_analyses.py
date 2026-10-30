"""Analyses added in response to the first-round (R1) peer review.

All flexible fits use the fixed modal random-forest setting with pre-experiment
predictors (F-pre) and study-balanced weights, and every comparison is paired
with the global first-order reference fitted on the identical calibration
studies. Preprocessing is fitted inside each calibration set by
``fixed_evaluate``; no evaluation study ever contributes to its own fit.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.modeling import fixed_evaluate
from src.models import ModelSpec
from src.robustness import evaluate_global_kinetic_baseline


def curve_mae(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error of every evaluated curve."""
    return (
        predictions.assign(absolute_error=(predictions["y_true"] - predictions["y_pred"]).abs())
        .groupby(["study_id", "curve_id"], sort=True)["absolute_error"]
        .mean()
        .rename("curve_mae")
        .reset_index()
    )


def study_curve_mae(predictions: pd.DataFrame) -> pd.Series:
    """Curve-balanced MAE of every evaluated study."""
    return curve_mae(predictions).groupby("study_id")["curve_mae"].mean()


def _rows(data: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    return np.flatnonzero(mask.to_numpy())


def _paired_fits(
    data: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: ModelSpec,
    parameters: dict,
    *,
    task: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the frozen forest and the global first-order reference on each fold."""
    forest, _ = fixed_evaluate(
        data,
        spec=spec,
        parameters=parameters,
        outer_folds=folds,
        task=task,
        weighting="study",
        random_state=random_state,
    )
    kinetic = evaluate_global_kinetic_baseline(data, folds, task=task, weighting="study")
    return forest, kinetic


# ---------------------------------------------------------------------------
# 1. Earlier withheld-instance validation versus complete-curve evaluation
# ---------------------------------------------------------------------------


def withheld_curve_ids(data: pd.DataFrame, withheld: pd.DataFrame) -> list[str]:
    """Map the original study/instance validation keys onto D3 curve identifiers."""
    keys = withheld.rename(columns={"excluded_instance": "instance"})[["study_id", "instance"]]
    curves = data.drop_duplicates("curve_id")[["study_id", "instance", "curve_id"]]
    return sorted(keys.merge(curves, on=["study_id", "instance"], how="inner")["curve_id"])


def withheld_instance_comparison(
    predictions: pd.DataFrame,
    withheld_curves: Sequence[str],
) -> pd.DataFrame:
    """Compare originally withheld curves with all remaining curves per task and model."""
    rows = []
    for (task, model), part in predictions.groupby(["task", "model"], sort=True):
        errors = curve_mae(part)
        flagged = errors["curve_id"].isin(withheld_curves)
        ranked = errors.sort_values("curve_mae").reset_index(drop=True)
        ranks = (ranked.index[ranked["curve_id"].isin(withheld_curves)] + 1).tolist()
        rows.append(
            {
                "task": task,
                "model": model,
                "withheld_curves": int(flagged.sum()),
                "remaining_curves": int((~flagged).sum()),
                "withheld_mean_curve_mae": float(errors.loc[flagged, "curve_mae"].mean()),
                "withheld_median_curve_mae": float(errors.loc[flagged, "curve_mae"].median()),
                "remaining_mean_curve_mae": float(errors.loc[~flagged, "curve_mae"].mean()),
                "remaining_median_curve_mae": float(errors.loc[~flagged, "curve_mae"].median()),
                "all_mean_curve_mae": float(errors["curve_mae"].mean()),
                "withheld_ranks_best_first": " ".join(map(str, ranks)),
            }
        )
    return pd.DataFrame(rows)


def withheld_design_distribution(
    data: pd.DataFrame,
    curve_errors: pd.DataFrame,
    *,
    min_curves_per_study: int = 4,
    min_points_per_curve: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate every validation set the earlier one-curve-per-study rule could draw.

    The rule withholds one curve with more than five observations from every
    study holding more than three curves. Each enumerated set is scored with the
    complete-curve out-of-fold errors supplied in ``curve_errors``.
    """
    points = data.groupby("curve_id").size()
    curves_per_study = data.groupby("study_id")["curve_id"].nunique()
    eligible_studies = sorted(curves_per_study.index[curves_per_study >= min_curves_per_study])
    options = []
    for study in eligible_studies:
        study_curves = sorted(data.loc[data["study_id"].eq(study), "curve_id"].unique())
        options.append([c for c in study_curves if points[c] >= min_points_per_curve])
    error_lookup = curve_errors.set_index("curve_id")["curve_mae"]
    draws = np.array(
        [np.mean([error_lookup[c] for c in combination]) for combination in itertools.product(*options)]
    )
    summary = pd.DataFrame(
        [
            {
                "eligible_studies": " ".join(map(str, eligible_studies)),
                "possible_validation_sets": int(len(draws)),
                "minimum_mean_curve_mae": float(draws.min()),
                "quantile_2_5": float(np.quantile(draws, 0.025)),
                "median": float(np.median(draws)),
                "quantile_97_5": float(np.quantile(draws, 0.975)),
                "maximum_mean_curve_mae": float(draws.max()),
                "all_curve_mean_mae": float(error_lookup.mean()),
            }
        ]
    )
    return summary, pd.DataFrame({"mean_curve_mae": draws})


# ---------------------------------------------------------------------------
# 2. Number of calibration studies
# ---------------------------------------------------------------------------


def study_count_learning_curve(
    data: pd.DataFrame,
    spec: ModelSpec,
    parameters: dict,
    *,
    study_counts: Sequence[int] = tuple(range(2, 11)),
    max_subsets: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Omitted-study error when calibration uses random subsets of k other studies."""
    studies = sorted(data["study_id"].unique())
    records = []
    for study_index, study in enumerate(studies):
        others = [s for s in studies if s != study]
        test_idx = _rows(data, data["study_id"].eq(study))
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        meta: list[tuple[int, int, tuple[int, ...]]] = []
        for k in study_counts:
            combinations = list(itertools.combinations(others, k))
            if len(combinations) > max_subsets:
                rng = np.random.default_rng(random_state + 1000 * int(study) + k)
                chosen = rng.choice(len(combinations), size=max_subsets, replace=False)
                combinations = [combinations[i] for i in sorted(chosen)]
            for subset_index, subset in enumerate(combinations):
                folds.append((_rows(data, data["study_id"].isin(subset)), test_idx))
                meta.append((k, subset_index, subset))
        forest, kinetic = _paired_fits(
            data, folds, spec, parameters, task="r1_study_count", random_state=random_state + 1000 * study_index
        )
        for model_name, frame in (("random_forest_f_pre_fixed", forest), ("global_first_order", kinetic)):
            for fold_index, part in frame.groupby("outer_fold", sort=True):
                k, subset_index, subset = meta[int(fold_index)]
                records.append(
                    {
                        "omitted_study": int(study),
                        "calibration_studies": k,
                        "subset_index": subset_index,
                        "subset": " ".join(map(str, subset)),
                        "model": model_name,
                        "study_curve_mae": float(curve_mae(part)["curve_mae"].mean()),
                    }
                )
        print(f"R1 study-count curve: omitted study {study} complete", flush=True)
    return pd.DataFrame(records)


def summarize_learning_curve(
    records: pd.DataFrame,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Equal-study summaries with study-bootstrap intervals for every k."""
    per_study = (
        records.groupby(["model", "calibration_studies", "omitted_study"])["study_curve_mae"]
        .mean()
        .unstack("model")
    )
    rng = np.random.default_rng(random_state)
    rows = []
    for k, part in per_study.groupby(level="calibration_studies"):
        part = part.droplevel("calibration_studies")
        difference = part["random_forest_f_pre_fixed"] - part["global_first_order"]
        draws = rng.integers(0, len(part), size=(bootstrap_iterations, len(part)))
        row = {"calibration_studies": int(k), "omitted_studies": int(len(part))}
        for column, values in (
            ("forest", part["random_forest_f_pre_fixed"]),
            ("kinetic", part["global_first_order"]),
            ("forest_minus_kinetic", difference),
        ):
            array = values.to_numpy(dtype=float)
            boot = array[draws].mean(axis=1)
            row[f"{column}_mae"] = float(array.mean())
            row[f"{column}_ci_lower"] = float(np.quantile(boot, 0.025))
            row[f"{column}_ci_upper"] = float(np.quantile(boot, 0.975))
        row["studies_forest_better"] = int((difference < 0).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    trend = spearmanr(summary["calibration_studies"], summary["forest_minus_kinetic_mae"])
    summary["trend_spearman_rho_difference"] = float(trend.statistic)
    summary["trend_spearman_p_difference"] = float(trend.pvalue)
    return summary


# ---------------------------------------------------------------------------
# 3. Environment-matched and reporting-complete calibration
# ---------------------------------------------------------------------------


def environment_matched_loso(
    data: pd.DataFrame,
    spec: ModelSpec,
    parameters: dict,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Omitted-study error with calibration restricted to the same receiving environment."""
    studies = sorted(data["study_id"].unique())
    parts = []
    fold_index = 0
    for study in studies:
        for environment in sorted(data.loc[data["study_id"].eq(study), "degradation_environment"].unique()):
            test_mask = data["study_id"].eq(study) & data["degradation_environment"].eq(environment)
            matched_mask = data["study_id"].ne(study) & data["degradation_environment"].eq(environment)
            all_mask = data["study_id"].ne(study)
            if data.loc[matched_mask, "study_id"].nunique() < 1:
                # No other study shares this environment, so no paired contrast exists.
                continue
            designs = {"all_other_studies": all_mask, "same_environment_studies": matched_mask}
            for design, train_mask in designs.items():
                folds = [(_rows(data, train_mask), _rows(data, test_mask))]
                forest, kinetic = _paired_fits(
                    data, folds, spec, parameters, task="r1_environment_matched", random_state=random_state + fold_index
                )
                fold_index += 1
                for model_name, frame in (("random_forest_f_pre_fixed", forest), ("global_first_order", kinetic)):
                    errors = curve_mae(frame)
                    errors["environment"] = environment
                    errors["calibration"] = design
                    errors["calibration_studies"] = int(data.loc[train_mask, "study_id"].nunique())
                    errors["model"] = model_name
                    parts.append(errors)
    return pd.concat(parts, ignore_index=True)


def reporting_complete_loso(
    data: pd.DataFrame,
    spec: ModelSpec,
    parameters: dict,
    complete_studies: Sequence[int],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Leave-one-study-out within studies that explicitly report blank correction and denominator."""
    parts = []
    for fold_index, study in enumerate(sorted(complete_studies)):
        test_mask = data["study_id"].eq(study)
        designs = {
            "reporting_complete_studies": data["study_id"].isin(complete_studies) & ~test_mask,
            "all_other_studies": ~test_mask,
        }
        for design_index, (design, train_mask) in enumerate(designs.items()):
            folds = [(_rows(data, train_mask), _rows(data, test_mask))]
            forest, kinetic = _paired_fits(
                data,
                folds,
                spec,
                parameters,
                task="r1_reporting_complete",
                random_state=random_state + 10 * fold_index + design_index,
            )
            for model_name, frame in (("random_forest_f_pre_fixed", forest), ("global_first_order", kinetic)):
                errors = curve_mae(frame)
                errors["calibration"] = design
                errors["calibration_studies"] = int(data.loc[train_mask, "study_id"].nunique())
                errors["model"] = model_name
                parts.append(errors)
    return pd.concat(parts, ignore_index=True)


def summarize_calibration_designs(curve_errors: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Equal-study, curve-balanced MAE for every calibration design and model."""
    keys = list(by) + ["calibration", "model"]
    per_study = curve_errors.groupby(keys + ["study_id"])["curve_mae"].mean().reset_index()
    return (
        per_study.groupby(keys)
        .agg(studies=("study_id", "nunique"), study_curve_mae=("curve_mae", "mean"))
        .reset_index()
    )


def reporting_completeness_association(
    per_study_mae: pd.DataFrame,
    descriptor_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate omitted-study error to explicit primary-text reporting of transfer descriptors."""
    descriptors = descriptor_matrix.set_index("study_id")
    explicit = descriptors.eq("R").sum(axis=1).rename("explicit_descriptors")
    applicable = descriptors.notna().sum(axis=1).rename("applicable_descriptors")
    joined = per_study_mae.join(explicit, on="study_id").join(applicable, on="study_id")
    joined["blank_and_denominator_explicit"] = joined["study_id"].map(
        descriptors["blank_correction"].eq("R") & descriptors["mineralization_denominator"].eq("R")
    )
    rows = []
    for column in [c for c in per_study_mae.columns if c != "study_id"]:
        rho = spearmanr(joined["explicit_descriptors"], joined[column])
        rows.append(
            {
                "quantity": column,
                "spearman_rho_with_explicit_descriptors": float(rho.statistic),
                "spearman_p": float(rho.pvalue),
                "median_blank_and_denominator_explicit": float(
                    joined.loc[joined["blank_and_denominator_explicit"], column].median()
                ),
                "median_otherwise": float(joined.loc[~joined["blank_and_denominator_explicit"], column].median()),
                "studies": int(len(joined)),
            }
        )
    return joined, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Bound on any advantage over the time-only reference
# ---------------------------------------------------------------------------


def reference_contrast_intervals(
    per_study: pd.DataFrame,
    candidates: Sequence[str],
    reference: str,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Study-level candidate-minus-reference MAE with 95% intervals.

    The t interval is the 95% interval of the Student-t posterior already used for
    the paired study-level diagnostics; the percentile interval resamples studies.
    Negative values favour the candidate, so the lower bound limits its possible advantage.
    """
    from scipy.stats import t as student_t

    from src.reviewer_analyses import paired_difference_diagnostics

    rng = np.random.default_rng(random_state)
    rows = []
    for candidate in candidates:
        differences = (per_study[candidate] - per_study[reference]).dropna()
        values = differences.to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean())
        scale = float(values.std(ddof=1)) / np.sqrt(n)
        low, high = student_t.ppf([0.025, 0.975], df=n - 1, loc=mean, scale=scale)
        draws = values[rng.integers(0, n, size=(bootstrap_iterations, n))].mean(axis=1)
        rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "candidate_mae": float(per_study[candidate].mean()),
                "reference_mae": float(per_study[reference].mean()),
                "t_interval_lower": float(low),
                "t_interval_upper": float(high),
                "bootstrap_lower": float(np.quantile(draws, 0.025)),
                "bootstrap_upper": float(np.quantile(draws, 0.975)),
                **paired_difference_diagnostics(values),
            }
        )
    return pd.DataFrame(rows)
