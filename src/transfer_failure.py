"""Kinetic-shape references and diagnostics for cross-study transfer failure.

The relationships in this module use elapsed time only.  Every fitted parameter,
including reference-family selection, is estimated inside the relevant calibration
partition so that an omitted study cannot influence its own reference curve.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from sklearn.isotonic import IsotonicRegression

from src.evaluation import regression_metrics
from src.experiments import summarize_predictions
from src.models import sample_weights
from src.splitting import IndexPair, curve_group_folds, leave_one_study_out_folds


REFERENCE_FAMILIES = (
    "first_order",
    "avrami_weibull",
    "gompertz",
    "logistic",
    "monotone_spline",
)


def _first_order(time: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    asymptote, rate = parameters
    return asymptote * (1.0 - np.exp(-rate * time))


def _avrami_weibull(time: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    asymptote, rate, exponent = parameters
    return asymptote * (1.0 - np.exp(-np.power(rate * time, exponent)))


def _gompertz(time: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    asymptote, rate, midpoint = parameters
    return asymptote * np.exp(-np.exp(-rate * (time - midpoint)))


def _logistic(time: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    asymptote, rate, midpoint = parameters
    argument = np.clip(-rate * (time - midpoint), -700.0, 700.0)
    return asymptote / (1.0 + np.exp(argument))


_PARAMETRIC = {
    "first_order": (_first_order, np.array([1e-6, 1e-8]), np.array([100.0, 10.0])),
    "avrami_weibull": (
        _avrami_weibull,
        np.array([1e-6, 1e-8, 0.10]),
        np.array([100.0, 10.0, 5.0]),
    ),
    "gompertz": (
        _gompertz,
        np.array([1e-6, 1e-8, -1.0e4]),
        np.array([100.0, 10.0, 1.0e4]),
    ),
    "logistic": (
        _logistic,
        np.array([1e-6, 1e-8, -1.0e4]),
        np.array([100.0, 10.0, 1.0e4]),
    ),
}


def _initial_values(family: str, time: np.ndarray, response: np.ndarray) -> list[np.ndarray]:
    asymptote = float(np.clip(max(np.nanpercentile(response, 90), 20.0), 20.0, 99.0))
    positive = time[time > 0]
    scale = float(np.nanmedian(positive)) if positive.size else 30.0
    rate = float(np.log(2.0) / max(scale, 1e-6))
    midpoint = float(np.nanmedian(time))
    if family == "first_order":
        return [np.array([asymptote, rate]), np.array([90.0, 0.02])]
    if family == "avrami_weibull":
        return [
            np.array([asymptote, rate, exponent])
            for exponent in (0.6, 1.0, 1.8)
        ]
    return [
        np.array([asymptote, rate, midpoint]),
        np.array([90.0, 0.02, scale]),
        np.array([90.0, 0.05, midpoint]),
    ]


def _fit_parametric(
    family: str,
    time: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
) -> tuple[callable, dict[str, object]]:
    relationship, lower, upper = _PARAMETRIC[family]
    root_weights = np.sqrt(weights / np.mean(weights))
    best = None
    for initial in _initial_values(family, time, response):
        initial = np.clip(initial, lower + 1e-10, upper - 1e-10)
        result = least_squares(
            lambda parameters: root_weights * (relationship(time, parameters) - response),
            x0=initial,
            bounds=(lower, upper),
            max_nfev=30_000,
        )
        loss = float(np.sum(np.square(result.fun)))
        if result.success and (best is None or loss < best[0]):
            best = (loss, result.x)
    if best is None:
        raise RuntimeError(f"{family} relationship did not converge")
    parameters = best[1]
    return (
        lambda new_time: np.clip(
            relationship(np.asarray(new_time, dtype=float), parameters), 0.0, 100.0
        ),
        {"parameters": [float(value) for value in parameters]},
    )


def _fit_monotone_spline(
    time: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
) -> tuple[callable, dict[str, object]]:
    frame = pd.DataFrame(
        {"x": np.log1p(time), "y": response, "weight": weights}
    )
    frame["weighted_y"] = frame["y"] * frame["weight"]
    grouped = frame.groupby("x", as_index=False, sort=True).agg(
        weighted_y=("weighted_y", "sum"),
        weight=("weight", "sum"),
    )
    grouped["y"] = grouped["weighted_y"] / grouped["weight"]
    x = grouped["x"].to_numpy(dtype=float)
    y = grouped["y"].to_numpy(dtype=float)
    weight = grouped["weight"].to_numpy(dtype=float)
    isotonic = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True)
    monotone = isotonic.fit_transform(x, y, sample_weight=weight)
    if len(x) == 1:
        value = float(monotone[0])
        return lambda new_time: np.repeat(value, len(np.atleast_1d(new_time))), {
            "knots": 1,
            "boundary": "constant",
        }
    spline = PchipInterpolator(x, monotone, extrapolate=True)

    def predict(new_time):
        new_x = np.log1p(np.asarray(new_time, dtype=float))
        bounded_x = np.clip(new_x, x.min(), x.max())
        return np.clip(spline(bounded_x), 0.0, 100.0)

    return predict, {
        "knots": int(len(x)),
        "boundary": "constant outside calibration-time range",
    }


def fit_reference_relationship(
    family: str,
    data: pd.DataFrame,
    *,
    weighting: str,
) -> tuple[callable, dict[str, object]]:
    """Fit one bounded, time-only reference relationship."""
    if family not in REFERENCE_FAMILIES:
        raise ValueError(f"Unknown reference family: {family}")
    time = data["biodegradation_time_days"].to_numpy(dtype=float)
    response = data["biodegradation_percentage"].to_numpy(dtype=float)
    weights = sample_weights(data, weighting)
    if family == "monotone_spline":
        return _fit_monotone_spline(time, response, weights)
    return _fit_parametric(family, time, response, weights)


def _prediction_frame(
    test: pd.DataFrame,
    test_indices: np.ndarray,
    prediction: np.ndarray,
    *,
    task: str,
    model: str,
    outer_fold: int,
    details: dict[str, object],
    weighting: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "task": task,
            "model": model,
            "outer_fold": outer_fold,
            "row_index": np.asarray(test_indices, dtype=int),
            "study_id": test["study_id"].to_numpy(),
            "curve_id": test["curve_id"].to_numpy(),
            "y_true": test["biodegradation_percentage"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
            "best_parameters": json.dumps(details, sort_keys=True),
            "weighting": weighting,
        }
    )


def evaluate_shape_references(
    data: pd.DataFrame,
    outer_folds: Iterable[IndexPair],
    *,
    task: str,
    weighting: str,
    inner_group_column: str,
    selection_metric: str,
    inner_splits: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed families and a training-only selected family in each outer fold."""
    prediction_parts: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for outer_fold, (train_index, test_index) in enumerate(outer_folds):
        train = data.iloc[train_index].copy()
        test = data.iloc[test_index].copy()
        inner_folds = curve_group_folds(
            train,
            n_splits=min(inner_splits, train[inner_group_column].nunique()),
            group_column=inner_group_column,
        )
        scores: dict[str, float] = {}
        for family in REFERENCE_FAMILIES:
            inner_parts = []
            for inner_fold, (inner_train_index, inner_valid_index) in enumerate(inner_folds):
                inner_train = train.iloc[inner_train_index]
                inner_valid = train.iloc[inner_valid_index]
                relationship, details = fit_reference_relationship(
                    family, inner_train, weighting=weighting
                )
                inner_parts.append(
                    _prediction_frame(
                        inner_valid,
                        inner_valid_index,
                        relationship(inner_valid["biodegradation_time_days"]),
                        task=task,
                        model=family,
                        outer_fold=inner_fold,
                        details=details,
                        weighting=weighting,
                    )
                )
            score = regression_metrics(pd.concat(inner_parts, ignore_index=True))[
                selection_metric
            ]
            scores[family] = float(score)
            tuning_rows.append(
                {
                    "task": task,
                    "outer_fold": outer_fold,
                    "reference_family": family,
                    "inner_selection_score": float(score),
                    "selection_metric": selection_metric,
                    "inner_group_column": inner_group_column,
                    "selected": False,
                }
            )
        selected = min(scores, key=lambda family: (scores[family], family))
        tuning_rows[-len(REFERENCE_FAMILIES) + REFERENCE_FAMILIES.index(selected)][
            "selected"
        ] = True
        fitted: dict[str, tuple[callable, dict[str, object]]] = {}
        for family in REFERENCE_FAMILIES:
            fitted[family] = fit_reference_relationship(family, train, weighting=weighting)
            relationship, details = fitted[family]
            prediction_parts.append(
                _prediction_frame(
                    test,
                    test_index,
                    relationship(test["biodegradation_time_days"]),
                    task=task,
                    model=family,
                    outer_fold=outer_fold,
                    details=details,
                    weighting=weighting,
                )
            )
        selected_relationship, selected_details = fitted[selected]
        prediction_parts.append(
            _prediction_frame(
                test,
                test_index,
                selected_relationship(test["biodegradation_time_days"]),
                task=task,
                model="selected_shape_reference",
                outer_fold=outer_fold,
                details={"selected_family": selected, **selected_details},
                weighting=weighting,
            )
        )
        audit_rows.append(
            {
                "task": task,
                "outer_fold": outer_fold,
                "selected_family": selected,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_studies": train["study_id"].nunique(),
                "test_studies": test["study_id"].nunique(),
                "study_overlap_count": len(set(train.study_id) & set(test.study_id)),
                "curve_overlap_count": len(set(train.curve_id) & set(test.curve_id)),
                "inner_group_column": inner_group_column,
                "selection_metric": selection_metric,
            }
        )
    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(tuning_rows),
        pd.DataFrame(audit_rows),
    )


def _curve_slope(part: pd.DataFrame, column: str) -> float:
    x = np.log1p(part["biodegradation_time_days"].to_numpy(dtype=float))
    y = part[column].to_numpy(dtype=float)
    if len(np.unique(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def phase_error_diagnostics(predictions: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Describe LOSO error by within-curve early, middle, and late thirds."""
    timing = []
    for curve_id, part in data.groupby("curve_id", sort=False):
        ordered = part.sort_values(["biodegradation_time_days", "point_index"])
        phases = np.array_split(np.arange(len(ordered)), 3)
        for label, positions in zip(("early", "middle", "late"), phases):
            for row_index in ordered.iloc[positions].index:
                timing.append({"row_index": int(row_index), "phase": label})
    timing_frame = pd.DataFrame(timing)
    merged = predictions.merge(timing_frame, on="row_index", how="left").merge(
        data[["biodegradation_time_days"]], left_on="row_index", right_index=True
    )
    rows = []
    for (model, phase), part in merged.groupby(["model", "phase"], sort=False):
        part = part.assign(error=part.y_pred - part.y_true)
        curve = part.groupby(["study_id", "curve_id"], sort=False).agg(
            mae=("error", lambda values: float(np.mean(np.abs(values)))),
            bias=("error", "mean"),
        )
        study = curve.groupby(level="study_id", sort=False).mean()
        slope_rows = []
        terminal_rows = []
        for (study_id, curve_id), curve_part in part.groupby(
            ["study_id", "curve_id"], sort=False
        ):
            slope_rows.append(
                {
                    "study_id": study_id,
                    "curve_id": curve_id,
                    "slope_error": abs(
                        _curve_slope(curve_part, "y_pred")
                        - _curve_slope(curve_part, "y_true")
                    ),
                }
            )
            last = curve_part.sort_values("biodegradation_time_days").iloc[-1]
            terminal_rows.append(
                {
                    "study_id": study_id,
                    "curve_id": curve_id,
                    "terminal_error": abs(float(last.y_pred - last.y_true)),
                }
            )
        slope = pd.DataFrame(slope_rows).groupby("study_id")["slope_error"].mean().mean()
        terminal = (
            pd.DataFrame(terminal_rows)
            .groupby("study_id")["terminal_error"]
            .mean()
            .mean()
        )
        rows.append(
            {
                "model": model,
                "phase": phase,
                "study_curve_macro_mae": float(study.mae.mean()),
                "study_curve_signed_bias": float(study.bias.mean()),
                "absolute_log_time_slope_error": float(slope),
                "terminal_absolute_error": float(terminal),
            }
        )
    return pd.DataFrame(rows)


def representative_studies(root: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    """Select best, median, worst, and most distant LOSO studies by fixed rules."""
    rf = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    rf = rf.loc[
        rf.task.eq("task_c_leave_one_study_out") & rf.model.eq("random_forest")
    ].assign(absolute_error=lambda frame: (frame.y_true - frame.y_pred).abs())
    curve = rf.groupby(["study_id", "curve_id"], as_index=False).absolute_error.mean()
    study = curve.groupby("study_id", as_index=False).absolute_error.mean()
    ordered = study.sort_values(["absolute_error", "study_id"]).reset_index(drop=True)
    chosen = {
        "lowest RF error": int(ordered.iloc[0].study_id),
        "median RF error": int(ordered.iloc[(len(ordered) - 1) // 2].study_id),
        "highest RF error": int(ordered.iloc[-1].study_id),
    }
    applicability = pd.read_csv(root / "results/audit/fold_local_applicability.csv")
    distance = (
        applicability.loc[applicability.task.eq("task_c_leave_one_study_out")]
        .groupby("study_id", as_index=False)
        .nearest_neighbor_distance.median()
        .sort_values(["nearest_neighbor_distance", "study_id"], ascending=[False, True])
    )
    distinct = next(
        int(value)
        for value in distance.study_id
        if int(value) not in set(chosen.values())
    )
    chosen["largest calibration-domain distance"] = distinct
    study_mae = study.set_index("study_id").absolute_error
    distance_value = distance.set_index("study_id").nearest_neighbor_distance
    rows = []
    for criterion, study_id in chosen.items():
        curve_part = curve.loc[curve.study_id.eq(study_id)].copy()
        target = float(curve_part.absolute_error.median())
        curve_part["distance_to_median"] = (curve_part.absolute_error - target).abs()
        curve_id = curve_part.sort_values(["distance_to_median", "curve_id"]).iloc[0].curve_id
        rows.append(
            {
                "criterion": criterion,
                "study_id": study_id,
                "representative_curve_id": curve_id,
                "rf_study_curve_macro_mae": float(study_mae.loc[study_id]),
                "median_calibration_domain_distance": float(distance_value.loc[study_id]),
                "curve_rule": "curve with RF MAE nearest the study median",
            }
        )
    return pd.DataFrame(rows)


def run_transfer_failure_analysis(
    root: Path,
    *,
    bootstrap_iterations: int = 5000,
) -> dict[str, pd.DataFrame]:
    """Run and persist the nested kinetic-shape transfer audit."""
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    data = data.reset_index(drop=True)
    designs = {
        "task_b_curve_grouped_nested_cv": {
            "folds": curve_group_folds(data, n_splits=5, group_column="curve_family_id"),
            "weighting": "curve",
            "inner_group_column": "curve_family_id",
            "selection_metric": "curve_macro_mae",
        },
        "task_c_leave_one_study_out": {
            "folds": leave_one_study_out_folds(data),
            "weighting": "study",
            "inner_group_column": "study_id",
            "selection_metric": "study_curve_macro_mae",
        },
    }
    prediction_parts = []
    tuning_parts = []
    audit_parts = []
    for task, design in designs.items():
        predictions, tuning, audit = evaluate_shape_references(
            data,
            design["folds"],
            task=task,
            weighting=design["weighting"],
            inner_group_column=design["inner_group_column"],
            selection_metric=design["selection_metric"],
        )
        prediction_parts.append(predictions)
        tuning_parts.append(tuning)
        audit_parts.append(audit)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    tuning = pd.concat(tuning_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    metrics = summarize_predictions(
        predictions,
        bootstrap_iterations=bootstrap_iterations,
        random_state=42,
    )

    strict = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    strict_c = strict.loc[
        strict.task.eq("task_c_leave_one_study_out")
        & strict.model.isin(["random_forest", "extra_trees", "xgboost"])
    ]
    phase_input = pd.concat(
        [
            strict_c,
            predictions.loc[
                predictions.task.eq("task_c_leave_one_study_out")
                & predictions.model.isin(
                    ["first_order", "selected_shape_reference", "monotone_spline"]
                )
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    phases = phase_error_diagnostics(phase_input, data)
    representatives = representative_studies(root, predictions)

    (root / "results/predictions").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    (root / "results/tuning").mkdir(parents=True, exist_ok=True)
    (root / "results/audit").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        root / "results/predictions/shape_reference_predictions.csv", index=False
    )
    metrics.to_csv(root / "results/metrics/shape_reference_metrics.csv", index=False)
    tuning.to_csv(root / "results/tuning/shape_reference_selection.csv", index=False)
    audit.to_csv(root / "results/audit/shape_reference_fold_audit.csv", index=False)
    phases.to_csv(root / "results/metrics/loso_phase_error_diagnostics.csv", index=False)
    representatives.to_csv(
        root / "results/audit/representative_loso_trajectories.csv", index=False
    )
    return {
        "predictions": predictions,
        "metrics": metrics,
        "tuning": tuning,
        "audit": audit,
        "phases": phases,
        "representatives": representatives,
    }


if __name__ == "__main__":
    run_transfer_failure_analysis(Path(__file__).resolve().parents[1])
