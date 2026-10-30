"""Precision/filtering sensitivity analysis with a frozen RF specification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation import grouped_mae_bootstrap_interval, regression_metrics
from src.modeling import fixed_evaluate
from src.models import model_catalog
from src.splitting import leave_one_study_out_folds


FROZEN_RF_PARAMETERS = {"max_depth": 12, "max_features": "sqrt", "min_samples_leaf": 6}


def _ensure_curve_id(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    if "curve_id" not in output:
        output["curve_id"] = output.apply(
            lambda row: f"S{int(row.study_id):02d}_I{int(row.instance):03d}", axis=1
        )
    return output


def _folds_from_curve_mapping(data: pd.DataFrame, curve_to_fold: dict[str, int]):
    assigned = data["curve_id"].map(curve_to_fold)
    if assigned.isna().any():
        raise ValueError("A sensitivity curve has no prespecified outer-fold assignment")
    folds = []
    for fold in sorted(assigned.unique()):
        test = assigned.index[assigned.eq(fold)].to_numpy(dtype=int)
        train = assigned.index[assigned.ne(fold)].to_numpy(dtype=int)
        folds.append((train, test))
    return folds


def run_dataset_sensitivity(
    root: Path, bootstrap_iterations: int = 5000, predictor_set: str = "F-all"
) -> dict[str, pd.DataFrame]:
    strict = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    reference = strict.loc[
        strict["task"].eq("task_b_curve_grouped_nested_cv")
        & strict["model"].eq("random_forest"),
        ["curve_id", "outer_fold"],
    ].drop_duplicates()
    if reference.groupby("curve_id")["outer_fold"].nunique().max() != 1:
        raise ValueError("Reference curve-fold mapping is not unique")
    curve_to_fold = reference.set_index("curve_id")["outer_fold"].to_dict()

    paths = {
        "D0_endpoint_audited_truncated": root / "data/processed/D0_published_cleaned.csv",
        "D1_endpoint_audited_exclude_gt100": root / "data/processed/D1_precision_preserved_exclude_gt100.csv",
        "D2_endpoint_audited_cap_gt100": root / "data/processed/D2_precision_preserved_cap_gt100.csv",
    }
    spec = model_catalog(predictor_set)["random_forest"]
    prediction_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    for dataset, path in paths.items():
        data = _ensure_curve_id(pd.read_csv(path))
        data = data[data["study_id"].isin(strict["study_id"].unique())].reset_index(drop=True)
        tasks = {
            "task_b_fixed_curve_folds": _folds_from_curve_mapping(data, curve_to_fold),
            "task_c_fixed_loso": leave_one_study_out_folds(data),
        }
        for task, folds in tasks.items():
            prediction, audit = fixed_evaluate(
                data,
                spec=spec,
                parameters=FROZEN_RF_PARAMETERS,
                outer_folds=folds,
                task=task,
                weighting="study" if task == "task_c_fixed_loso" else "curve",
                random_state=42,
            )
            prediction.insert(0, "dataset", dataset)
            audit.insert(0, "dataset", dataset)
            prediction_parts.append(prediction)
            audit_parts.append(audit)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    summary_rows = []
    for index, ((dataset, task), part) in enumerate(predictions.groupby(["dataset", "task"], sort=False)):
        group_level = "study" if task == "task_c_fixed_loso" else "curve"
        interval = grouped_mae_bootstrap_interval(
            part,
            group_level=group_level,
            iterations=bootstrap_iterations,
            random_state=1200 + index,
        )
        primary_metric = (
            "study_curve_macro_mae" if group_level == "study" else "curve_macro_mae"
        )
        summary_rows.append(
            {"dataset": dataset, "task": task, "model": "random_forest_frozen"}
            | regression_metrics(part)
            | {
                "curve_macro_mae_ci_lower": interval["lower"],
                "curve_macro_mae_ci_upper": interval["upper"],
                "primary_mae_metric": primary_metric,
                "primary_mae": interval["estimate"],
                "primary_mae_ci_lower": interval["lower"],
                "primary_mae_ci_upper": interval["upper"],
                "rows": len(part),
                "curves": part["curve_id"].nunique(),
                "studies": part["study_id"].nunique(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    (root / "results/predictions").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    (root / "results/audit").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(root / "results/predictions/dataset_sensitivity_predictions.csv", index=False)
    summary.to_csv(root / "results/metrics/dataset_sensitivity_metrics.csv", index=False)
    audit.to_csv(root / "results/audit/dataset_sensitivity_fold_audit.csv", index=False)
    return {"predictions": predictions, "summary": summary, "audit": audit}


if __name__ == "__main__":
    run_dataset_sensitivity(Path(__file__).resolve().parents[1])
