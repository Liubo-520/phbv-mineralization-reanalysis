"""Recompute analyses requested during statistical and deployment review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.endpoint_contamination import (
    endpoint_contamination_predictions,
    random_row_contamination_predictions,
    summarize_endpoint_contamination,
)
from src.experiments import summarize_predictions
from src.modeling import fixed_evaluate, nested_evaluate
from src.reviewer_analyses import (
    canonical_group_codes,
    leave_one_unit_out_influence,
    loso_unknown_category_audit,
    paired_difference_diagnostics,
    predictor_feature_sets,
    predictor_provenance_table,
    profile_group_ids,
    study_loss_differences,
    task_b_interval_sensitivity,
    task_b_study_representation_audit,
)
from src.sensitivity import FROZEN_RF_PARAMETERS
from src.splitting import curve_group_folds, leave_one_study_out_folds


def _ensure_output_directories(root: Path) -> None:
    for relative in (
        "results/predictions",
        "results/metrics",
        "results/tuning",
        "results/audit",
        "tables/supplementary",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def run_predictor_availability_analysis(
    root: Path,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Nested Task B/C evaluation of conservative ex-ante and full features."""
    _ensure_output_directories(root)
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    prediction_parts = []
    tuning_parts = []
    audit_parts = []
    task_designs = {
        "task_b_curve_grouped_nested_cv": {
            "folds": curve_group_folds(data, n_splits=5, group_column="curve_family_id"),
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
    for feature_set, spec in predictor_feature_sets().items():
        for task, design in task_designs.items():
            prediction, tuning, audit = nested_evaluate(
                data,
                spec=spec,
                outer_folds=design["folds"],
                task=task,
                inner_splits=4,
                inner_group_column=design["inner_group_column"],
                selection_metric=design["selection_metric"],
                weighting=design["weighting"],
                random_state=random_state,
            )
            model_label = f"random_forest_{feature_set.lower().replace('-', '_')}"
            prediction["model_family"] = prediction["model"]
            prediction["model"] = model_label
            prediction["feature_set"] = feature_set
            tuning["feature_set"] = feature_set
            audit["feature_set"] = feature_set
            prediction_parts.append(prediction)
            tuning_parts.append(tuning)
            audit_parts.append(audit)
            print(f"predictor availability: {feature_set} / {task} complete", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    fold_audit = pd.concat(audit_parts, ignore_index=True)
    metrics = summarize_predictions(
        predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    intervals = task_b_interval_sensitivity(
        predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    unknown_categories = loso_unknown_category_audit(
        data, predictor_feature_sets()["F-all"].categorical_features
    )
    provenance = predictor_provenance_table(data)

    comparison_rows = []
    influence_parts = []
    for task in task_designs:
        differences = study_loss_differences(
            predictions,
            task=task,
            candidate_model="random_forest_f_pre",
            reference_model="random_forest_f_all",
        )
        comparison_rows.append(
            {
                "task": task,
                "candidate": "F-pre",
                "reference": "F-all",
                **paired_difference_diagnostics(differences.to_numpy()),
            }
        )
        influence = leave_one_unit_out_influence(differences)
        influence.insert(0, "task", task)
        influence_parts.append(influence)
    comparisons = pd.DataFrame(comparison_rows)
    influence = pd.concat(influence_parts, ignore_index=True)

    predictions.to_csv(
        root / "results/predictions/predictor_availability_predictions.csv", index=False
    )
    tuning.to_csv(root / "results/tuning/predictor_availability_tuning.csv", index=False)
    fold_audit.to_csv(
        root / "results/audit/predictor_availability_fold_audit.csv", index=False
    )
    metrics.to_csv(root / "results/metrics/predictor_availability_metrics.csv", index=False)
    intervals.to_csv(
        root / "results/metrics/predictor_availability_task_b_intervals.csv", index=False
    )
    comparisons.to_csv(
        root / "results/metrics/predictor_availability_comparisons.csv", index=False
    )
    influence.to_csv(
        root / "results/audit/predictor_availability_leave_one_study_out.csv", index=False
    )
    unknown_categories.to_csv(
        root / "results/audit/predictor_loso_unknown_categories.csv", index=False
    )
    provenance.to_csv(
        root / "tables/supplementary/Table_S6_predictor_provenance.csv", index=False
    )
    return {
        "predictions": predictions,
        "tuning": tuning,
        "fold_audit": fold_audit,
        "metrics": metrics,
        "intervals": intervals,
        "comparisons": comparisons,
        "influence": influence,
        "unknown_categories": unknown_categories,
        "provenance": provenance,
    }


def run_fpre_deployment_validation(
    root: Path,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Evaluate the exact frozen public RF using only ex-ante predictors."""
    _ensure_output_directories(root)
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    spec = predictor_feature_sets()["F-pre"]
    designs = {
        "task_b_curve_grouped_nested_cv": (
            curve_group_folds(data, n_splits=5, group_column="curve_family_id"),
            "curve",
        ),
        "task_c_leave_one_study_out": (leave_one_study_out_folds(data), "study"),
    }
    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for task, (folds, weighting) in designs.items():
        predictions, audit = fixed_evaluate(
            data,
            spec=spec,
            parameters=FROZEN_RF_PARAMETERS,
            outer_folds=folds,
            task=task,
            weighting=weighting,
            random_state=random_state,
        )
        predictions["model_family"] = predictions["model"]
        predictions["model"] = "random_forest_f_pre_fixed"
        predictions["feature_set"] = "F-pre"
        audit["feature_set"] = "F-pre"
        prediction_parts.append(predictions)
        audit_parts.append(audit)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    metrics = summarize_predictions(
        predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    predictions.to_csv(
        root / "results/predictions/fpre_deployment_predictions.csv", index=False
    )
    metrics.to_csv(root / "results/metrics/fpre_deployment_metrics.csv", index=False)
    audit.to_csv(root / "results/audit/fpre_deployment_fold_audit.csv", index=False)
    return {"predictions": predictions, "metrics": metrics, "audit": audit}


def run_profile_grouping_sensitivity(
    root: Path,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
    predictor_set: str = "F-all",
) -> dict[str, pd.DataFrame]:
    """Evaluate frozen RF under alternate exact-profile definitions."""
    _ensure_output_directories(root)
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    spec = predictor_feature_sets()[predictor_set]
    static_features = tuple(
        feature
        for feature in spec.numeric_features + spec.categorical_features
        if feature != "biodegradation_time_days"
    )
    variants = {
        "study_exact": data["curve_family_id"].astype(str),
        "profile_no_study_exact": profile_group_ids(
            data, static_features, include_study=False
        ),
        "profile_no_study_tolerance_0_5pp": profile_group_ids(
            data,
            static_features,
            include_study=False,
            continuous_tolerance=0.5,
        ),
    }
    prediction_parts = []
    fold_audit_parts = []
    representation_parts = []
    definition_rows = []
    for variant, group_ids in variants.items():
        group_ids = canonical_group_codes(group_ids)
        variant_data = data.copy()
        variant_data["profile_group_id"] = group_ids
        variant_data["curve_family_id"] = group_ids
        folds = curve_group_folds(
            variant_data, n_splits=5, group_column="profile_group_id"
        )
        prediction, audit = fixed_evaluate(
            variant_data,
            spec=spec,
            parameters=FROZEN_RF_PARAMETERS,
            outer_folds=folds,
            task="task_b_curve_grouped_nested_cv",
            weighting="curve",
            random_state=random_state,
        )
        prediction["model_family"] = prediction["model"]
        prediction["model"] = f"random_forest_{variant}"
        prediction["grouping_variant"] = variant
        audit["grouping_variant"] = variant
        representation = task_b_study_representation_audit(
            variant_data, folds, grouping_variant=variant
        )
        prediction_parts.append(prediction)
        fold_audit_parts.append(audit)
        representation_parts.append(representation)
        definition_rows.append(
            {
                "grouping_variant": variant,
                "profile_groups": int(group_ids.nunique()),
                "curves": int(data["curve_id"].nunique()),
                "groups_with_multiple_curves": int(
                    data.assign(profile_group_id=group_ids)
                    .drop_duplicates("curve_id")["profile_group_id"]
                    .value_counts()
                    .gt(1)
                    .sum()
                ),
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_audit = pd.concat(fold_audit_parts, ignore_index=True)
    representation = pd.concat(representation_parts, ignore_index=True)
    definitions = pd.DataFrame(definition_rows)
    metrics = pd.concat(
        [
            summarize_predictions(
                part,
                bootstrap_iterations=bootstrap_iterations,
                random_state=random_state,
            )
            for _, part in predictions.groupby("grouping_variant", sort=False)
        ],
        ignore_index=True,
    )
    predictions.to_csv(
        root / "results/predictions/profile_grouping_sensitivity_predictions.csv",
        index=False,
    )
    metrics.to_csv(
        root / "results/metrics/profile_grouping_sensitivity_metrics.csv", index=False
    )
    fold_audit.to_csv(
        root / "results/audit/profile_grouping_fold_audit.csv", index=False
    )
    representation.to_csv(
        root / "results/audit/task_b_study_representation.csv", index=False
    )
    definitions.to_csv(
        root / "results/audit/profile_grouping_definitions.csv", index=False
    )
    return {
        "predictions": predictions,
        "metrics": metrics,
        "fold_audit": fold_audit,
        "representation": representation,
        "definitions": definitions,
    }


def run_paired_inference_audit(root: Path) -> dict[str, pd.DataFrame]:
    """Compute exact, influence, and Bayesian study-level paired diagnostics."""
    _ensure_output_directories(root)
    strict = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    kinetic = pd.read_csv(
        root / "results/predictions/global_kinetic_baseline_predictions.csv"
    )
    comparisons = []
    influence_parts = []
    paired_loss_parts = []

    task_c = pd.concat(
        [
            strict.loc[
                strict["task"].eq("task_c_leave_one_study_out")
                & strict["model"].eq("random_forest")
            ],
            kinetic.loc[
                kinetic["task"].eq("task_c_leave_one_study_out")
                & kinetic["model"].eq("global_first_order")
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    task_pairs = [
        (
            task_c,
            "task_c_leave_one_study_out",
            "random_forest",
            "global_first_order",
        )
    ]
    fpre_path = root / "results/predictions/fpre_deployment_predictions.csv"
    if fpre_path.is_file():
        fpre = pd.read_csv(fpre_path)
        fpre_task_c = pd.concat(
            [
                fpre.loc[fpre["task"].eq("task_c_leave_one_study_out")],
                kinetic.loc[
                    kinetic["task"].eq("task_c_leave_one_study_out")
                    & kinetic["model"].eq("global_first_order")
                ],
            ],
            ignore_index=True,
            sort=False,
        )
        task_pairs.append(
            (
                fpre_task_c,
                "task_c_leave_one_study_out",
                "random_forest_f_pre_fixed",
                "global_first_order",
            )
        )
    task_b = strict.loc[strict["task"].eq("task_b_curve_grouped_nested_cv")]
    for candidate in sorted(set(task_b["model"]).difference({"random_forest"})):
        task_pairs.append(
            (
                task_b,
                "task_b_curve_grouped_nested_cv",
                candidate,
                "random_forest",
            )
        )

    for predictions, task, candidate, reference in task_pairs:
        differences = study_loss_differences(
            predictions,
            task=task,
            candidate_model=candidate,
            reference_model=reference,
        )
        comparisons.append(
            {
                "task": task,
                "candidate": candidate,
                "reference": reference,
                **paired_difference_diagnostics(differences.to_numpy()),
            }
        )
        influence = leave_one_unit_out_influence(differences)
        influence.insert(0, "reference", reference)
        influence.insert(0, "candidate", candidate)
        influence.insert(0, "task", task)
        influence_parts.append(influence)
        paired_losses = differences.rename("mae_difference").reset_index()
        paired_losses.insert(0, "reference", reference)
        paired_losses.insert(0, "candidate", candidate)
        paired_losses.insert(0, "task", task)
        paired_loss_parts.append(paired_losses)

    diagnostics = pd.DataFrame(comparisons)
    influence = pd.concat(influence_parts, ignore_index=True)
    paired_losses = pd.concat(paired_loss_parts, ignore_index=True)
    diagnostics.to_csv(
        root / "results/metrics/exact_paired_study_diagnostics.csv", index=False
    )
    influence.to_csv(
        root / "results/audit/paired_leave_one_study_out_influence.csv", index=False
    )
    paired_losses.to_csv(
        root / "results/metrics/paired_study_mae_differences.csv", index=False
    )
    return {
        "diagnostics": diagnostics,
        "influence": influence,
        "paired_losses": paired_losses,
    }


def run_endpoint_contamination_analysis(
    root: Path,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
    predictor_set: str = "F-all",
) -> dict[str, pd.DataFrame]:
    """Evaluate negative transfer from mass-loss endpoint contamination."""
    _ensure_output_directories(root)
    co2 = pd.read_csv(
        root / "data/processed/D3_endpoint_audited_mineralization.csv"
    )
    all_endpoints = pd.read_csv(
        root / "data/processed/D1_precision_preserved_exclude_gt100.csv"
    )
    co2["endpoint_class"] = "co2_mineralization"
    mass_loss = all_endpoints.loc[all_endpoints["study_id"].isin([5, 6])].copy()
    mass_loss["endpoint_class"] = "mass_loss"
    spec = predictor_feature_sets()[predictor_set]

    loso_predictions, loso_audit = endpoint_contamination_predictions(
        co2,
        mass_loss,
        spec=spec,
        parameters=FROZEN_RF_PARAMETERS,
        random_state=random_state,
    )
    random_predictions, random_audit = random_row_contamination_predictions(
        co2,
        mass_loss,
        spec=spec,
        parameters=FROZEN_RF_PARAMETERS,
        random_state=random_state,
    )
    predictions = pd.concat(
        [loso_predictions, random_predictions], ignore_index=True, sort=False
    )
    metrics, comparisons = summarize_endpoint_contamination(
        predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    predictions.to_csv(
        root / "results/predictions/endpoint_contamination_predictions.csv", index=False
    )
    metrics.to_csv(
        root / "results/metrics/endpoint_contamination_metrics.csv", index=False
    )
    comparisons.to_csv(
        root / "results/metrics/endpoint_contamination_comparisons.csv", index=False
    )
    loso_audit.to_csv(
        root / "results/audit/endpoint_contamination_loso_audit.csv", index=False
    )
    random_audit.to_csv(
        root / "results/audit/endpoint_contamination_random_row_audit.csv", index=False
    )
    return {
        "predictions": predictions,
        "metrics": metrics,
        "comparisons": comparisons,
        "loso_audit": loso_audit,
        "random_audit": random_audit,
    }


def run_all_reviewer_analyses(
    root: Path,
    *,
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
    predictor_set: str = "F-all",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run and persist every reviewer-requested statistical sensitivity."""
    return {
        "predictor_availability": run_predictor_availability_analysis(
            root,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
        ),
        "fpre_deployment": run_fpre_deployment_validation(
            root,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
        ),
        "profile_grouping": run_profile_grouping_sensitivity(
            root,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
            predictor_set=predictor_set,
        ),
        "paired_inference": run_paired_inference_audit(root),
        "endpoint_contamination": run_endpoint_contamination_analysis(
            root,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
            predictor_set=predictor_set,
        ),
    }
