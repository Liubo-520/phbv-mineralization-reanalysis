"""Feature, weighting, and kinetic-baseline robustness analyses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.evaluation import grouped_mae_bootstrap_interval, regression_metrics
from src.modeling import fixed_evaluate
from src.models import POST_EXPERIMENT_FEATURES, ModelSpec, model_catalog, sample_weights
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.sensitivity import FROZEN_RF_PARAMETERS
from src.splitting import IndexPair, curve_group_folds, leave_one_study_out_folds
from src.statistics import paired_curve_comparisons, paired_study_comparisons


def feature_ablation_specs() -> dict[str, ModelSpec]:
    environmental_numeric = ("biodegradation_time_days", "t_biodeg")
    environmental_categorical = (
        "biodegradation_condition",
        "degradation_environment",
        "degradation_mechanism",
        "pha_degrading_microbes",
    )
    material_numeric = (
        "biodegradation_time_days",
        "adjusted_hb_ratio_formulation_mol",
        "adjusted_hv_ratio_formulation_mol",
        "additive1_percentage_wt",
        "additive2_percentage_wt",
        "additive3_percentage_wt",
    )
    material_categorical = (
        "additives",
        "additive_type_1",
        "additive_type_2",
        "additive_type_3",
        "sample_shape_morphology",
    )
    candidate = (FROZEN_RF_PARAMETERS,)
    return {
        "F0_time": ModelSpec("random_forest", ("biodegradation_time_days",), (), candidate),
        "F1_time_environment": ModelSpec(
            "random_forest", environmental_numeric, environmental_categorical, candidate
        ),
        "F2_time_material": ModelSpec(
            "random_forest", material_numeric, material_categorical, candidate
        ),
        "F3_full": ModelSpec(
            "random_forest", tuple(NUMERIC_FEATURES), tuple(CATEGORICAL_FEATURES), candidate
        ),
    }


def _first_order(time: np.ndarray, asymptote: float, rate: float) -> np.ndarray:
    return asymptote * (1.0 - np.exp(-rate * time))


def time_forward_split(
    data: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    minimum_unique_times: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split eligible curves at a strict time boundary for within-curve extrapolation."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if minimum_unique_times < 3:
        raise ValueError("minimum_unique_times must be at least three")
    work = data.reset_index(drop=True)
    train_positions: list[int] = []
    test_positions: list[int] = []
    audit_rows: list[dict] = []
    for curve_id, part in work.groupby("curve_id", sort=False):
        times = np.sort(part["biodegradation_time_days"].dropna().unique())
        if len(times) < minimum_unique_times:
            continue
        train_time_count = min(len(times) - 1, max(2, int(np.floor(len(times) * train_fraction))))
        boundary = float(times[train_time_count - 1])
        curve_train = part.index[part["biodegradation_time_days"].le(boundary)].to_numpy()
        curve_test = part.index[part["biodegradation_time_days"].gt(boundary)].to_numpy()
        if not len(curve_train) or not len(curve_test):
            continue
        train_positions.extend(map(int, curve_train))
        test_positions.extend(map(int, curve_test))
        audit_rows.append(
            {
                "curve_id": curve_id,
                "train_rows": len(curve_train),
                "test_rows": len(curve_test),
                "train_max_time": boundary,
                "test_min_time": float(work.loc[curve_test, "biodegradation_time_days"].min()),
            }
        )
    if not audit_rows:
        raise ValueError("No curves satisfy the time-forward eligibility rule")
    audit = pd.DataFrame(audit_rows)
    audit["eligible_curves"] = len(audit)
    audit["excluded_curves"] = int(work["curve_id"].nunique() - len(audit))
    audit["train_fraction_rule"] = train_fraction
    audit["minimum_unique_times"] = minimum_unique_times
    return (
        np.asarray(train_positions, dtype=int),
        np.asarray(test_positions, dtype=int),
        audit,
    )


def curve_time_forward_baselines(
    data: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.DataFrame:
    """Predict held-out late times from each curve's own early observations."""
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    parts: list[pd.DataFrame] = []
    for curve_id, test_curve in test.groupby("curve_id", sort=False):
        train_curve = train.loc[train["curve_id"].eq(curve_id)].sort_values(
            "biodegradation_time_days"
        )
        x_train = train_curve["biodegradation_time_days"].to_numpy(dtype=float)
        y_train = train_curve["biodegradation_percentage"].to_numpy(dtype=float)
        x_test = test_curve["biodegradation_time_days"].to_numpy(dtype=float)
        last_prediction = np.repeat(y_train[-1], len(test_curve))
        initial_asymptote = float(np.clip(max(y_train.max(), 1.0), 1e-6, 100.0))
        fit = least_squares(
            lambda parameters: _first_order(x_train, *parameters) - y_train,
            x0=np.array([initial_asymptote, 0.02]),
            bounds=(np.array([1e-6, 1e-8]), np.array([100.0, 10.0])),
            max_nfev=20_000,
        )
        kinetic_prediction = _first_order(x_test, *fit.x)
        test_positions = test_curve.index.to_numpy(dtype=int)
        common = {
            "task": "task_d_time_forward",
            "outer_fold": 0,
            "row_index": test_positions,
            "study_id": test_curve["study_id"].to_numpy(),
            "curve_id": test_curve["curve_id"].to_numpy(),
            "y_true": test_curve["biodegradation_percentage"].to_numpy(dtype=float),
            "weighting": "curve",
        }
        parts.append(
            pd.DataFrame(
                common
                | {
                    "model": "last_observation",
                    "y_pred": np.clip(last_prediction, 0.0, 100.0),
                    "best_parameters": "{}",
                }
            )
        )
        parts.append(
            pd.DataFrame(
                common
                | {
                    "model": "curve_first_order",
                    "y_pred": np.clip(kinetic_prediction, 0.0, 100.0),
                    "best_parameters": (
                        f'{{"asymptote": {fit.x[0]:.12g}, "rate": {fit.x[1]:.12g}}}'
                    ),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_time_forward_analysis(
    data: pd.DataFrame, predictor_set: str = "F-all"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the secondary, within-trajectory future-time extrapolation task."""
    train_idx, test_idx, audit = time_forward_split(data)
    rf_predictions, _ = fixed_evaluate(
        data,
        spec=model_catalog(predictor_set)["random_forest"],
        parameters=FROZEN_RF_PARAMETERS,
        outer_folds=[(train_idx, test_idx)],
        task="task_d_time_forward",
        weighting="curve",
    )
    rf_predictions["model"] = "random_forest"
    predictions = pd.concat(
        [rf_predictions, curve_time_forward_baselines(data, train_idx, test_idx)],
        ignore_index=True,
    )
    summary = _summaries(predictions, ["task", "model"])
    return predictions, summary, audit


def evaluate_global_kinetic_baseline(
    data: pd.DataFrame,
    outer_folds: Iterable[IndexPair],
    *,
    task: str,
    weighting: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for outer_fold, (train_idx, test_idx) in enumerate(outer_folds):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        x = train["biodegradation_time_days"].to_numpy(dtype=float)
        y = train["biodegradation_percentage"].to_numpy(dtype=float)
        weights = sample_weights(train, weighting)

        def residual(parameters: np.ndarray) -> np.ndarray:
            return np.sqrt(weights) * (_first_order(x, *parameters) - y)

        fit = least_squares(
            residual,
            x0=np.array([90.0, 0.02]),
            bounds=(np.array([1e-6, 1e-8]), np.array([100.0, 10.0])),
            max_nfev=20_000,
        )
        prediction = np.clip(
            _first_order(test["biodegradation_time_days"].to_numpy(dtype=float), *fit.x),
            0.0,
            100.0,
        )
        parts.append(
            pd.DataFrame(
                {
                    "task": task,
                    "model": "global_first_order",
                    "outer_fold": outer_fold,
                    "row_index": np.asarray(test_idx, dtype=int),
                    "study_id": test["study_id"].to_numpy(),
                    "curve_id": test["curve_id"].to_numpy(),
                    "y_true": test["biodegradation_percentage"].to_numpy(dtype=float),
                    "y_pred": prediction,
                    "best_parameters": f'{{"asymptote": {fit.x[0]:.12g}, "rate": {fit.x[1]:.12g}}}',
                    "weighting": weighting,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _summaries(
    predictions: pd.DataFrame,
    group_columns: list[str],
    *,
    bootstrap_iterations: int = 5000,
) -> pd.DataFrame:
    rows = []
    for group_index, (keys, part) in enumerate(predictions.groupby(group_columns, sort=False)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identifiers = dict(zip(group_columns, keys))
        task = str(identifiers.get("task", ""))
        group_level = "study" if ("task_c" in task or "loso" in task) else "curve"
        primary_metric = (
            "study_curve_macro_mae" if group_level == "study" else "curve_macro_mae"
        )
        metrics = regression_metrics(part)
        interval = grouped_mae_bootstrap_interval(
            part,
            group_level=group_level,
            iterations=bootstrap_iterations,
            random_state=4200 + group_index,
        )
        rows.append(
            identifiers
            | metrics
            | {
                "rows": len(part),
                "curves": part["curve_id"].nunique(),
                "studies": part["study_id"].nunique(),
                "primary_mae_metric": primary_metric,
                "primary_mae": metrics[primary_metric],
                "primary_mae_ci_lower": interval["lower"],
                "primary_mae_ci_upper": interval["upper"],
            }
        )
    return pd.DataFrame(rows)


def run_robustness_suite(root: Path, predictor_set: str = "F-all") -> dict[str, pd.DataFrame]:
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    task_designs = {
        "task_b_curve_grouped_nested_cv": {
            "folds": curve_group_folds(data, n_splits=5, group_column="curve_family_id"),
            "primary_weighting": "curve",
        },
        "task_c_leave_one_study_out": {
            "folds": leave_one_study_out_folds(data),
            "primary_weighting": "study",
        },
    }

    ablation_parts = []
    for feature_set, spec in feature_ablation_specs().items():
        if predictor_set == "F-pre":
            spec = replace(
                spec,
                categorical_features=tuple(
                    feature
                    for feature in spec.categorical_features
                    if feature not in POST_EXPERIMENT_FEATURES
                ),
            )
        for task, design in task_designs.items():
            prediction, _ = fixed_evaluate(
                data,
                spec=spec,
                parameters=FROZEN_RF_PARAMETERS,
                outer_folds=design["folds"],
                task=task,
                weighting=design["primary_weighting"],
            )
            prediction["model"] = f"random_forest_{feature_set}"
            prediction["feature_set"] = feature_set
            ablation_parts.append(prediction)
            print(f"robustness: {task} / {feature_set} complete", flush=True)
    ablation_predictions = pd.concat(ablation_parts, ignore_index=True)
    ablation_summary = _summaries(ablation_predictions, ["task", "feature_set"])

    full_spec = model_catalog(predictor_set)["random_forest"]
    weighting_parts = []
    for weighting in ("point", "curve", "study"):
        for task, design in task_designs.items():
            prediction, _ = fixed_evaluate(
                data,
                spec=full_spec,
                parameters=FROZEN_RF_PARAMETERS,
                outer_folds=design["folds"],
                task=task,
                weighting=weighting,
            )
            prediction["model"] = f"random_forest_{weighting}_weighted"
            prediction["weighting"] = weighting
            weighting_parts.append(prediction)
            print(f"robustness: {task} / {weighting} weighting complete", flush=True)
    weighting_predictions = pd.concat(weighting_parts, ignore_index=True)
    weighting_summary = _summaries(weighting_predictions, ["task", "weighting"])

    kinetic_parts = []
    for task, design in task_designs.items():
        kinetic_parts.append(
            evaluate_global_kinetic_baseline(
                data,
                design["folds"],
                task=task,
                weighting=design["primary_weighting"],
            )
        )
    kinetic_predictions = pd.concat(kinetic_parts, ignore_index=True)
    kinetic_summary = _summaries(kinetic_predictions, ["task", "model"])
    time_forward_predictions, time_forward_summary, time_forward_audit = (
        run_time_forward_analysis(data, predictor_set)
    )
    strict_predictions = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    strict_with_kinetic = pd.concat(
        [strict_predictions, kinetic_predictions], ignore_index=True, sort=False
    )
    primary_fixed = weighting_predictions.loc[
        (
            weighting_predictions["task"].eq("task_b_curve_grouped_nested_cv")
            & weighting_predictions["weighting"].eq("curve")
        )
        | (
            weighting_predictions["task"].eq("task_c_leave_one_study_out")
            & weighting_predictions["weighting"].eq("study")
        )
    ]
    fixed_with_kinetic = pd.concat(
        [primary_fixed, kinetic_predictions], ignore_index=True, sort=False
    )

    (root / "results/predictions").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    ablation_predictions.to_csv(root / "results/predictions/feature_ablation_predictions.csv", index=False)
    ablation_summary.to_csv(root / "results/metrics/feature_ablation_metrics.csv", index=False)
    weighting_predictions.to_csv(root / "results/predictions/weighting_sensitivity_predictions.csv", index=False)
    weighting_summary.to_csv(root / "results/metrics/weighting_sensitivity_metrics.csv", index=False)
    kinetic_predictions.to_csv(root / "results/predictions/global_kinetic_baseline_predictions.csv", index=False)
    kinetic_summary.to_csv(root / "results/metrics/global_kinetic_baseline_metrics.csv", index=False)
    time_forward_predictions.to_csv(
        root / "results/predictions/time_forward_predictions.csv", index=False
    )
    time_forward_summary.to_csv(
        root / "results/metrics/time_forward_metrics.csv", index=False
    )
    time_forward_audit.to_csv(
        root / "results/audit/time_forward_split_audit.csv", index=False
    )
    paired_curve_comparisons(strict_with_kinetic, "global_first_order").to_csv(
        root / "results/metrics/paired_curve_global_baseline_comparisons.csv", index=False
    )
    paired_study_comparisons(strict_with_kinetic, "global_first_order").to_csv(
        root / "results/metrics/paired_study_global_baseline_comparisons.csv", index=False
    )
    paired_curve_comparisons(fixed_with_kinetic, "global_first_order").to_csv(
        root / "results/metrics/paired_curve_frozen_global_comparisons.csv", index=False
    )
    paired_study_comparisons(fixed_with_kinetic, "global_first_order").to_csv(
        root / "results/metrics/paired_study_frozen_global_comparisons.csv", index=False
    )
    paired_curve_comparisons(time_forward_predictions, "last_observation").to_csv(
        root / "results/metrics/paired_curve_time_forward_comparisons.csv", index=False
    )
    return {
        "ablation_predictions": ablation_predictions,
        "ablation_summary": ablation_summary,
        "weighting_predictions": weighting_predictions,
        "weighting_summary": weighting_summary,
        "kinetic_predictions": kinetic_predictions,
        "kinetic_summary": kinetic_summary,
        "time_forward_predictions": time_forward_predictions,
        "time_forward_summary": time_forward_summary,
        "time_forward_audit": time_forward_audit,
    }


if __name__ == "__main__":
    run_robustness_suite(Path(__file__).resolve().parents[1])
