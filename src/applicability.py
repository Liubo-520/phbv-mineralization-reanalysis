"""Fold-local novelty diagnostics for mixed-variable model applicability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from src.models import categorical_features_for
from src.preprocessing import NUMERIC_FEATURES, build_preprocessor


def novelty_flags(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    numeric_count = np.zeros(len(test), dtype=int)
    for feature in numeric_features:
        lower = train[feature].min()
        upper = train[feature].max()
        numeric_count += (test[feature].lt(lower) | test[feature].gt(upper)).fillna(False).to_numpy(dtype=int)
    categorical_count = np.zeros(len(test), dtype=int)
    for feature in categorical_features:
        seen = set(train[feature].dropna())
        categorical_count += (~test[feature].isin(seen)).to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "numeric_out_of_range_count": numeric_count,
            "unseen_category_count": categorical_count,
        },
        index=test.index,
    )


def other_group_nearest_distances(matrix, groups: np.ndarray) -> np.ndarray:
    """Distance from each row to its nearest row belonging to another group."""
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two groups are required")
    neighbor = NearestNeighbors(n_neighbors=len(groups), metric="euclidean").fit(matrix)
    distances, indices = neighbor.kneighbors(matrix, return_distance=True)
    output = np.empty(len(groups), dtype=float)
    for row in range(len(groups)):
        different = groups[indices[row]] != groups[row]
        output[row] = distances[row, np.flatnonzero(different)[0]]
    return output


def _holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_applicability_analysis(root: Path, predictor_set: str = "F-all") -> dict[str, pd.DataFrame]:
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    predictions = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    predictions = predictions[predictions["model"].eq("random_forest")].copy()
    categorical = list(categorical_features_for(predictor_set))
    features = NUMERIC_FEATURES + categorical
    parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    for task, task_predictions in predictions.groupby("task", sort=False):
        for outer_fold, fold_predictions in task_predictions.groupby("outer_fold", sort=False):
            test_idx = fold_predictions["row_index"].to_numpy(dtype=int)
            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            preprocessor = build_preprocessor(NUMERIC_FEATURES, categorical)
            x_train = preprocessor.fit_transform(train[features])
            x_test = preprocessor.transform(test[features])
            neighbor = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(x_train)
            train_distances = other_group_nearest_distances(x_train, train["curve_id"].to_numpy())
            threshold = float(np.quantile(train_distances, 0.95))
            test_distances = neighbor.kneighbors(x_test, n_neighbors=1, return_distance=True)[0][:, 0]
            flags = novelty_flags(train, test, NUMERIC_FEATURES, categorical).reset_index(drop=True)
            fold = fold_predictions.sort_values("row_index").reset_index(drop=True).copy()
            ordered_test = test.reset_index(names="row_index").sort_values("row_index").reset_index(drop=True)
            if not np.array_equal(fold["row_index"], ordered_test["row_index"]):
                raise ValueError("Prediction and novelty row alignment failed")
            fold["nearest_neighbor_distance"] = test_distances
            fold["fold_distance_threshold_95"] = threshold
            fold["beyond_distance_domain"] = test_distances > threshold
            fold["numeric_out_of_range_count"] = flags["numeric_out_of_range_count"]
            fold["unseen_category_count"] = flags["unseen_category_count"]
            fold["absolute_error"] = (fold["y_true"] - fold["y_pred"]).abs()
            parts.append(fold)
            audit_rows.append(
                {
                    "task": task,
                    "outer_fold": outer_fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "distance_threshold_95": threshold,
                    "test_rows_beyond_distance_domain": int(fold["beyond_distance_domain"].sum()),
                    "test_rows_with_unseen_category": int(fold["unseen_category_count"].gt(0).sum()),
                    "test_rows_with_numeric_extrapolation": int(fold["numeric_out_of_range_count"].gt(0).sum()),
                }
            )

    novelty = pd.concat(parts, ignore_index=True)
    correlation_rows: list[dict] = []
    aggregated_parts: list[pd.DataFrame] = []
    for task, part in novelty.groupby("task", sort=False):
        levels = {
            "point": part[["nearest_neighbor_distance", "absolute_error"]],
            "curve": part.groupby(["study_id", "curve_id"], as_index=False).agg(
                nearest_neighbor_distance=("nearest_neighbor_distance", "median"),
                absolute_error=("absolute_error", "mean"),
            ),
            "study": part.groupby("study_id", as_index=False).agg(
                nearest_neighbor_distance=("nearest_neighbor_distance", "median"),
                absolute_error=("absolute_error", "mean"),
            ),
        }
        for level, aggregate in levels.items():
            rho, p_value = spearmanr(aggregate["nearest_neighbor_distance"], aggregate["absolute_error"])
            correlation_rows.append(
                {"task": task, "level": level, "clusters": len(aggregate), "spearman_rho": rho, "p_value": p_value}
            )
            aggregate = aggregate.copy()
            aggregate["task"] = task
            aggregate["level"] = level
            aggregated_parts.append(aggregate)
    correlations = pd.DataFrame(correlation_rows)
    correlations["p_value_holm"] = _holm_adjust(correlations["p_value"].tolist())

    bins = []
    for task, part in novelty.groupby("task", sort=False):
        part = part.copy()
        part["distance_quintile"] = pd.qcut(
            part["nearest_neighbor_distance"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop"
        )
        bins.append(
            part.groupby("distance_quintile", observed=True)
            .agg(rows=("absolute_error", "size"), curves=("curve_id", "nunique"), mean_absolute_error=("absolute_error", "mean"), median_distance=("nearest_neighbor_distance", "median"))
            .reset_index()
            .assign(task=task)
        )
    distance_bins = pd.concat(bins, ignore_index=True)
    audit = pd.DataFrame(audit_rows)

    (root / "results/audit").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    novelty.to_csv(root / "results/audit/fold_local_applicability.csv", index=False)
    audit.to_csv(root / "results/audit/applicability_fold_summary.csv", index=False)
    correlations.to_csv(root / "results/metrics/applicability_error_correlations.csv", index=False)
    distance_bins.to_csv(root / "results/metrics/applicability_distance_bins.csv", index=False)
    return {"novelty": novelty, "audit": audit, "correlations": correlations, "distance_bins": distance_bins}


if __name__ == "__main__":
    run_applicability_analysis(Path(__file__).resolve().parents[1])
