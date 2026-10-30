"""Recompute the R1 analysis with pre-experiment predictors (F-pre) as the primary set.

The initial-submission artifacts under ``results/`` are left untouched. The R1
analysis is materialized in ``analysis_r1/``, which receives byte-identical
copies of the shared inputs and then reruns the complete strict pipeline with
``predictor_set="F-pre"`` before the reviewer-requested analyses are added.

Usage::

    python -m src.r1_runner --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R1_ROOT = ROOT / "analysis_r1"
PRIMARY_PREDICTOR_SET = "F-pre"
R1_RESULTS = "results/r1_reviewer"

SHARED_INPUTS = (
    "data/processed/D0_published_cleaned.csv",
    "data/processed/D1_precision_preserved_exclude_gt100.csv",
    "data/processed/D2_precision_preserved_cap_gt100.csv",
    "data/processed/D3_endpoint_audited_mineralization.csv",
    "data/processed/D_raw_unbounded.csv",
    "data/audit/original_split_audit.csv",
    "data/audit/original_validation_curves.csv",
    "references/source_endpoint_audit.csv",
    "results/metrics/original_reproduction_metrics.csv",
    "results/predictions/original_held_out_curve_predictions.csv",
    "results/audit/study_descriptor_audit_long.csv",
    "results/audit/study_descriptor_completeness_by_study.csv",
    "results/audit/study_descriptor_completeness_by_variable.csv",
    "tables/supplementary/Table_S8_study_descriptor_matrix.csv",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_r1_root(root: Path = ROOT, r1_root: Path = R1_ROOT) -> Path:
    """Copy the shared inputs into the R1 analysis root and verify byte identity."""
    for relative in SHARED_INPUTS:
        source = root / relative
        target = r1_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha256(source) != _sha256(target):
            raise RuntimeError(f"R1 input copy differs from its source: {relative}")
    return r1_root


def run_r1_primary_pipeline(r1_root: Path = R1_ROOT, *, bootstrap_iterations: int = 5000) -> None:
    """Rerun the strict pipeline and its diagnostics with F-pre as the primary set."""
    from src.applicability import run_applicability_analysis
    from src.experiments import run_strict_experiments
    from src.interpretability import run_grouped_permutation_importance
    from src.kinetics import run_kinetic_analysis
    from src.physical import run_physical_analysis
    from src.reviewer_analysis_runner import run_all_reviewer_analyses
    from src.robustness import run_robustness_suite
    from src.sensitivity import run_dataset_sensitivity
    from src.statistics import paired_curve_comparisons, paired_study_comparisons
    from src.transfer_failure import run_transfer_failure_analysis
    from src.uncertainty import run_grouped_conformal

    predictor_set = PRIMARY_PREDICTOR_SET
    d3 = r1_root / "data/processed/D3_endpoint_audited_mineralization.csv"
    run_strict_experiments(
        d3,
        r1_root / "results",
        bootstrap_iterations=bootstrap_iterations,
        predictor_set=predictor_set,
    )
    strict = pd.read_csv(r1_root / "results/predictions/strict_oof_predictions.csv")
    paired_curve_comparisons(strict, reference_model="random_forest").to_csv(
        r1_root / "results/metrics/paired_curve_model_comparisons.csv", index=False
    )
    paired_study_comparisons(strict, reference_model="random_forest").to_csv(
        r1_root / "results/metrics/paired_study_model_comparisons.csv", index=False
    )
    run_dataset_sensitivity(r1_root, bootstrap_iterations, predictor_set=predictor_set)
    run_physical_analysis(
        d3,
        r1_root / "results/predictions/strict_oof_predictions.csv",
        r1_root / "results",
    )
    run_grouped_conformal(r1_root, predictor_set=predictor_set)
    run_kinetic_analysis(r1_root)
    run_applicability_analysis(r1_root, predictor_set=predictor_set)
    run_grouped_permutation_importance(r1_root, predictor_set=predictor_set)
    run_robustness_suite(r1_root, predictor_set=predictor_set)
    run_all_reviewer_analyses(
        r1_root,
        bootstrap_iterations=bootstrap_iterations,
        random_state=42,
        predictor_set=predictor_set,
    )
    run_transfer_failure_analysis(r1_root, bootstrap_iterations=bootstrap_iterations)
    print("R1 primary F-pre pipeline complete.", flush=True)


def run_r1_reviewer_analyses(r1_root: Path = R1_ROOT) -> dict[str, pd.DataFrame]:
    """Add the analyses requested in the first-round reviews."""
    from src.r1_analyses import (
        curve_mae,
        environment_matched_loso,
        reporting_complete_loso,
        reporting_completeness_association,
        study_count_learning_curve,
        study_curve_mae,
        summarize_calibration_designs,
        summarize_learning_curve,
        withheld_curve_ids,
        withheld_design_distribution,
        withheld_instance_comparison,
    )
    from src.reviewer_analyses import predictor_feature_sets
    from src.sensitivity import FROZEN_RF_PARAMETERS

    out = r1_root / R1_RESULTS
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(r1_root / "data/processed/D3_endpoint_audited_mineralization.csv")
    spec = predictor_feature_sets()[PRIMARY_PREDICTOR_SET]
    strict = pd.read_csv(r1_root / "results/predictions/strict_oof_predictions.csv")
    availability = pd.read_csv(r1_root / "results/predictions/predictor_availability_predictions.csv")
    kinetic = pd.read_csv(r1_root / "results/predictions/global_kinetic_baseline_predictions.csv")
    outputs: dict[str, pd.DataFrame] = {}

    # Earlier withheld-instance validation versus complete-curve evaluation.
    withheld = withheld_curve_ids(data, pd.read_csv(r1_root / "data/audit/original_validation_curves.csv"))
    nested = pd.concat(
        [
            strict.loc[strict["model"].eq("random_forest")].assign(model="random_forest_f_pre"),
            availability.loc[availability["model"].eq("random_forest_f_all")],
        ],
        ignore_index=True,
        sort=False,
    )
    original = pd.read_csv(r1_root / "results/predictions/original_held_out_curve_predictions.csv")
    original = original.loc[original["curve_id"].isin(withheld)]
    original_frame = pd.DataFrame(
        {
            "task": "earlier_withheld_instance_validation",
            "model": "earlier_tuned_rf",
            "study_id": original["study_id"],
            "curve_id": original["curve_id"],
            "y_true": original["biodegradation_percentage"],
            "y_pred": original["y_pred_described_tuned_rf"],
        }
    )
    outputs["withheld_instance_comparison"] = pd.concat(
        [withheld_instance_comparison(nested, withheld), withheld_instance_comparison(original_frame, withheld)],
        ignore_index=True,
    )
    task_b_rf = curve_mae(
        strict.loc[strict["task"].eq("task_b_curve_grouped_nested_cv") & strict["model"].eq("random_forest")]
    )
    design_summary, design_draws = withheld_design_distribution(data, task_b_rf)
    design_summary["earlier_co2_withheld_mean_curve_mae"] = float(
        task_b_rf.loc[task_b_rf["curve_id"].isin(withheld), "curve_mae"].mean()
    )
    design_summary["earlier_co2_withheld_percentile"] = float(
        (design_draws["mean_curve_mae"] <= design_summary.loc[0, "earlier_co2_withheld_mean_curve_mae"]).mean()
        * 100.0
    )
    outputs["withheld_design_summary"] = design_summary
    outputs["withheld_design_draws"] = design_draws

    # Number of calibration studies.
    records = study_count_learning_curve(data, spec, FROZEN_RF_PARAMETERS)
    outputs["study_count_records"] = records
    outputs["study_count_summary"] = summarize_learning_curve(records)

    # Environment-matched calibration.
    environment = environment_matched_loso(data, spec, FROZEN_RF_PARAMETERS)
    outputs["environment_matched_curves"] = environment
    outputs["environment_matched_summary"] = pd.concat(
        [
            summarize_calibration_designs(environment).assign(environment="all_matched_environments"),
            summarize_calibration_designs(environment, by=["environment"]),
        ],
        ignore_index=True,
        sort=False,
    )

    # Reporting-complete calibration.
    descriptors = pd.read_csv(r1_root / "tables/supplementary/Table_S8_study_descriptor_matrix.csv")
    complete = sorted(
        descriptors.loc[
            descriptors["blank_correction"].eq("R") & descriptors["mineralization_denominator"].eq("R"),
            "study_id",
        ].astype(int)
    )
    reporting = reporting_complete_loso(data, spec, FROZEN_RF_PARAMETERS, complete)
    outputs["reporting_complete_curves"] = reporting
    reporting_summary = summarize_calibration_designs(reporting)
    random_four = (
        records.loc[records["calibration_studies"].eq(len(complete) - 1) & records["omitted_study"].isin(complete)]
        .groupby(["model", "omitted_study"])["study_curve_mae"]
        .mean()
        .groupby("model")
        .mean()
    )
    for model, value in random_four.items():
        reporting_summary.loc[len(reporting_summary)] = {
            "calibration": f"random_{len(complete) - 1}_study_subsets",
            "model": model,
            "studies": len(complete),
            "study_curve_mae": float(value),
        }
    reporting_summary["reporting_complete_studies"] = " ".join(map(str, complete))
    outputs["reporting_complete_summary"] = reporting_summary

    # Omitted-study error versus explicit reporting.
    task_c = strict.loc[strict["task"].eq("task_c_leave_one_study_out") & strict["model"].eq("random_forest")]
    kinetic_c = kinetic.loc[kinetic["task"].eq("task_c_leave_one_study_out")]
    per_study = pd.DataFrame(
        {
            "forest_mae": study_curve_mae(task_c),
            "kinetic_mae": study_curve_mae(kinetic_c),
        }
    )
    per_study["forest_minus_kinetic"] = per_study["forest_mae"] - per_study["kinetic_mae"]
    joined, association = reporting_completeness_association(
        per_study.reset_index().rename(columns={"index": "study_id"}), descriptors
    )
    outputs["reporting_study_errors"] = joined
    outputs["reporting_association"] = association

    for name, frame in outputs.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    print("R1 reviewer analyses complete.", flush=True)
    return outputs


def run_r1_summaries(r1_root: Path = R1_ROOT) -> pd.DataFrame:
    """Study-level contrasts of every flexible relationship with the first-order reference."""
    from src.r1_analyses import reference_contrast_intervals, study_curve_mae

    task = "task_c_leave_one_study_out"
    results = r1_root / "results/predictions"
    strict = pd.read_csv(results / "strict_oof_predictions.csv")
    sources = {
        f"{model}_f_pre": strict[strict["model"].eq(model)]
        for model in ("ridge", "random_forest", "extra_trees", "xgboost")
    }
    fixed = pd.read_csv(results / "fpre_deployment_predictions.csv")
    sources["random_forest_f_pre_fixed"] = fixed
    availability = pd.read_csv(results / "predictor_availability_predictions.csv")
    sources["random_forest_f_all"] = availability[availability["model"].eq("random_forest_f_all")]
    kinetic = pd.read_csv(results / "global_kinetic_baseline_predictions.csv")
    sources["global_first_order"] = kinetic
    per_study = pd.DataFrame(
        {name: study_curve_mae(frame[frame["task"].eq(task)]) for name, frame in sources.items()}
    )
    contrasts = reference_contrast_intervals(
        per_study, [name for name in sources if name != "global_first_order"], "global_first_order"
    )
    contrasts.to_csv(r1_root / R1_RESULTS / "kinetic_reference_contrasts.csv", index=False)
    print("R1 summaries complete.", flush=True)
    return contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "primary", "reviewer", "summaries", "all"), default="all"
    )
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare_r1_root()
        print("R1 inputs prepared.", flush=True)
    if args.stage in ("primary", "all"):
        run_r1_primary_pipeline()
    if args.stage in ("reviewer", "all"):
        run_r1_reviewer_analyses()
    if args.stage in ("summaries", "all"):
        run_r1_summaries()


if __name__ == "__main__":
    main()
