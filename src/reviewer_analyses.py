"""Reviewer-requested predictor, grouping, and paired-inference audits."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

from src.evaluation import grouped_mae_bootstrap_interval, hierarchical_bootstrap_interval
from src.models import POST_EXPERIMENT_FEATURES, ModelSpec, model_catalog


SOURCE_CURATED_POST_EXPERIMENT_FEATURES = set(POST_EXPERIMENT_FEATURES)


PROVENANCE = {
    "biodegradation_time_days": (
        "Biodegradation_data time coordinate",
        "directly reported or digitized",
        "yes",
    ),
    "adjusted_hb_ratio_formulation_mol": (
        "Materials_features formulation fields",
        "derived from reported formulation",
        "yes",
    ),
    "adjusted_hv_ratio_formulation_mol": (
        "Materials_features formulation fields",
        "derived from reported formulation",
        "yes",
    ),
    "t_biodeg": (
        "Environmental_features incubation temperature",
        "directly reported protocol setting",
        "yes",
    ),
    "additive1_percentage_wt": (
        "Materials_features additive package",
        "directly reported or normalized",
        "yes",
    ),
    "additive2_percentage_wt": (
        "Materials_features additive package",
        "directly reported or normalized",
        "yes",
    ),
    "additive3_percentage_wt": (
        "Materials_features additive package",
        "directly reported or normalized",
        "yes",
    ),
    "biodegradation_condition": (
        "Environmental_features test protocol",
        "source-curated protocol label",
        "yes",
    ),
    "degradation_mechanism": (
        "Materials_features *Degradation_mechanism",
        "source-curated mechanistic interpretation",
        "not reliably",
    ),
    "additives": (
        "Materials_features additive fields",
        "derived presence flag",
        "yes",
    ),
    "degradation_environment": (
        "Environmental_features test medium",
        "source-curated normalized protocol label",
        "yes",
    ),
    "additive_type_1": (
        "Materials_features additive package",
        "directly reported or normalized",
        "yes",
    ),
    "additive_type_2": (
        "Materials_features additive package",
        "directly reported or normalized",
        "yes",
    ),
    "additive_type_3": (
        "Materials_features additive package",
        "directly reported or normalized",
        "yes",
    ),
    "sample_shape_morphology": (
        "Materials_features specimen description",
        "source-curated normalized material label",
        "yes",
    ),
    "pha_degrading_microbes": (
        "Environmental_features {**PHA_degrading_microbes}",
        "source-curated organism/activity label",
        "not reliably",
    ),
}


def predictor_feature_sets() -> dict[str, ModelSpec]:
    """Return the full and conservative ex-ante random-forest specifications."""
    full = model_catalog()["random_forest"]
    pre_categories = tuple(
        feature
        for feature in full.categorical_features
        if feature not in SOURCE_CURATED_POST_EXPERIMENT_FEATURES
    )
    return {
        "F-pre": ModelSpec(
            name="random_forest",
            numeric_features=full.numeric_features,
            categorical_features=pre_categories,
            candidates=full.candidates,
        ),
        "F-all": full,
    }


def loso_unknown_category_audit(
    data: pd.DataFrame,
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    """Count held-out categories absent from each LOSO training partition."""
    rows: list[dict[str, object]] = []
    for held_out_study in sorted(data["study_id"].unique()):
        train = data.loc[~data["study_id"].eq(held_out_study)]
        test = data.loc[data["study_id"].eq(held_out_study)]
        for feature in categorical_features:
            train_values = set(train[feature].dropna().astype(str))
            test_values = test[feature].dropna().astype(str)
            unknown = sorted(set(test_values).difference(train_values))
            rows.append(
                {
                    "held_out_study": held_out_study,
                    "feature": feature,
                    "unknown_categories": "|".join(unknown),
                    "unknown_category_count": len(unknown),
                    "unknown_test_rows": int(test_values.isin(unknown).sum()),
                    "test_rows": int(len(test)),
                }
            )
    return pd.DataFrame(rows)


def predictor_provenance_table(data: pd.DataFrame) -> pd.DataFrame:
    """Describe source, curation, ex-ante availability, and LOSO category novelty."""
    feature_sets = predictor_feature_sets()
    full = feature_sets["F-all"]
    categorical = list(full.categorical_features)
    unknown = loso_unknown_category_audit(data, categorical)
    rows = []
    for variable in full.numeric_features + full.categorical_features:
        source, provenance, availability = PROVENANCE[variable]
        part = unknown.loc[
            unknown["feature"].eq(variable) & unknown["unknown_category_count"].gt(0)
        ]
        unknown_studies = "|".join(map(str, part["held_out_study"].tolist()))
        rows.append(
            {
                "variable": variable,
                "original_source": source,
                "provenance_class": provenance,
                "known_before_experiment": availability,
                "feature_set": (
                    "F-all only"
                    if variable in SOURCE_CURATED_POST_EXPERIMENT_FEATURES
                    else "F-pre and F-all"
                ),
                "loso_unknown_studies": unknown_studies,
                "loso_has_unknown_category": bool(unknown_studies),
            }
        )
    return pd.DataFrame(rows)


def _profile_value(value: object, *, tolerance: float | None) -> str:
    if pd.isna(value):
        return "<NA>"
    if tolerance is not None and isinstance(value, (int, float, np.integer, np.floating)):
        return f"{round(float(value) / tolerance) * tolerance:.12g}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    return str(value)


def profile_group_ids(
    data: pd.DataFrame,
    static_features: Sequence[str],
    *,
    include_study: bool,
    continuous_tolerance: float | None = None,
) -> pd.Series:
    """Hash static profiles with optional study identity and numeric tolerance."""
    if continuous_tolerance is not None and continuous_tolerance <= 0:
        raise ValueError("continuous_tolerance must be positive")
    columns = (["study_id"] if include_study else []) + list(static_features)
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")

    identifiers = []
    for values in data[columns].itertuples(index=False, name=None):
        signature = "\x1f".join(
            _profile_value(value, tolerance=continuous_tolerance) for value in values
        )
        identifiers.append(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16])
    return pd.Series(identifiers, index=data.index, name="profile_group_id")


def canonical_group_codes(groups: pd.Series) -> pd.Series:
    """Encode an equivalence relation by first appearance, independent of labels."""
    codes, _ = pd.factorize(groups, sort=False)
    identifiers = [f"group_{code:04d}" for code in codes]
    return pd.Series(identifiers, index=groups.index, name="profile_group_id")


def paired_difference_diagnostics(differences: Sequence[float]) -> dict[str, float | int]:
    """Exact and Bayesian summaries for candidate-minus-reference paired losses."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("At least one finite paired difference is required")

    nonzero = values[values != 0]
    positive = int(np.sum(nonzero > 0))
    negative = int(np.sum(nonzero < 0))
    sign_p = (
        float(binomtest(positive, n=len(nonzero), p=0.5, alternative="two-sided").pvalue)
        if len(nonzero)
        else 1.0
    )

    if len(values) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 paired units")
    observed = abs(float(values.mean()))
    permuted = np.fromiter(
        (
            abs(float(np.mean(values * np.asarray(signs, dtype=float))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ),
        dtype=float,
    )
    permutation_p = float(np.mean(permuted >= observed - 1e-12))

    mean = float(values.mean())
    if len(values) == 1:
        posterior_probability = 1.0 if mean > 0 else 0.0 if mean < 0 else 0.5
    else:
        standard_deviation = float(values.std(ddof=1))
        if standard_deviation == 0:
            posterior_probability = 1.0 if mean > 0 else 0.0 if mean < 0 else 0.5
        else:
            standard_error = standard_deviation / np.sqrt(len(values))
            posterior_probability = float(
                1.0 - t.cdf(0.0, df=len(values) - 1, loc=mean, scale=standard_error)
            )

    return {
        "independent_units": int(len(values)),
        "candidate_worse_units": positive,
        "candidate_better_units": negative,
        "tied_units": int(np.sum(values == 0)),
        "mean_difference": mean,
        "median_difference": float(np.median(values)),
        "exact_sign_p_two_sided": sign_p,
        "exact_permutation_p_two_sided": permutation_p,
        "posterior_probability_mean_gt_zero": posterior_probability,
    }


def leave_one_unit_out_influence(differences: pd.Series) -> pd.DataFrame:
    """Recompute paired summaries after deleting each independent unit once."""
    finite = differences.astype(float).loc[np.isfinite(differences.astype(float))]
    if len(finite) < 2:
        raise ValueError("At least two finite paired differences are required")
    rows = []
    for omitted in finite.index:
        retained = finite.drop(index=omitted)
        rows.append(
            {
                "omitted_unit": omitted,
                "retained_units": int(len(retained)),
                "mean_difference": float(retained.mean()),
                "median_difference": float(retained.median()),
            }
        )
    return pd.DataFrame(rows)


def task_b_interval_sensitivity(
    predictions: pd.DataFrame,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Contrast curve-conditional and study-clustered Task B uncertainty."""
    task_b = predictions.loc[predictions["task"].eq("task_b_curve_grouped_nested_cv")]
    rows = []
    for model_index, (model, part) in enumerate(task_b.groupby("model", sort=False)):
        designs = [
            (
                "curve bootstrap conditional on represented studies",
                "curve_macro_mae",
                grouped_mae_bootstrap_interval(
                    part,
                    group_level="curve",
                    iterations=bootstrap_iterations,
                    random_state=random_state + model_index,
                ),
            ),
            (
                "hierarchical bootstrap: study then curve",
                "curve_macro_mae",
                hierarchical_bootstrap_interval(
                    part,
                    metric="curve_macro_mae",
                    iterations=bootstrap_iterations,
                    random_state=random_state + 1000 + model_index,
                ),
            ),
            (
                "study-clustered bootstrap of study-curve MAE",
                "study_curve_macro_mae",
                hierarchical_bootstrap_interval(
                    part,
                    metric="study_curve_macro_mae",
                    iterations=bootstrap_iterations,
                    random_state=random_state + 2000 + model_index,
                ),
            ),
        ]
        for design, estimand, interval in designs:
            rows.append(
                {
                    "task": "task_b_curve_grouped_nested_cv",
                    "model": model,
                    "resampling_design": design,
                    "estimand": estimand,
                    "estimate": interval["estimate"],
                    "ci_lower": interval["lower"],
                    "ci_upper": interval["upper"],
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )
    return pd.DataFrame(rows)


def task_b_study_representation_audit(
    data: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    grouping_variant: str,
) -> pd.DataFrame:
    """Record whether each Task B test study also contributes training curves."""
    rows = []
    for outer_fold, (train_index, test_index) in enumerate(folds):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        for study_id in sorted(test["study_id"].unique()):
            train_curves = train.loc[train["study_id"].eq(study_id), "curve_id"].nunique()
            test_curves = test.loc[test["study_id"].eq(study_id), "curve_id"].nunique()
            rows.append(
                {
                    "grouping_variant": grouping_variant,
                    "outer_fold": outer_fold,
                    "test_study": study_id,
                    "test_curves": int(test_curves),
                    "training_curves_same_study": int(train_curves),
                    "study_represented_in_training": bool(train_curves > 0),
                }
            )
    return pd.DataFrame(rows)


def study_loss_differences(
    predictions: pd.DataFrame,
    *,
    task: str,
    candidate_model: str,
    reference_model: str,
) -> pd.Series:
    """Return candidate-minus-reference curve-balanced MAE for each study."""
    part = predictions.loc[
        predictions["task"].eq(task)
        & predictions["model"].isin([candidate_model, reference_model])
    ].copy()
    part["absolute_error"] = np.abs(part["y_true"] - part["y_pred"])
    curve_losses = (
        part.groupby(["model", "study_id", "curve_id"], sort=False)["absolute_error"]
        .mean()
        .reset_index()
    )
    study_losses = (
        curve_losses.groupby(["model", "study_id"], sort=False)["absolute_error"]
        .mean()
        .unstack("model")
    )
    required = {candidate_model, reference_model}
    if not required.issubset(study_losses.columns):
        raise ValueError(f"Missing model losses: {sorted(required.difference(study_losses.columns))}")
    return (study_losses[candidate_model] - study_losses[reference_model]).rename(
        "mae_difference"
    )
