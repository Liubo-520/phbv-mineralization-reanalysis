"""Prespecified traditional-ML model families and weighting rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


POST_EXPERIMENT_FEATURES = frozenset({"degradation_mechanism", "pha_degrading_microbes"})
PREDICTOR_SETS = ("F-all", "F-pre")


def categorical_features_for(predictor_set: str) -> tuple[str, ...]:
    """Return the categorical predictors of one predictor-availability set."""
    if predictor_set == "F-all":
        return tuple(CATEGORICAL_FEATURES)
    if predictor_set == "F-pre":
        return tuple(
            feature for feature in CATEGORICAL_FEATURES if feature not in POST_EXPERIMENT_FEATURES
        )
    raise ValueError(f"Unknown predictor set: {predictor_set}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]


def curve_balanced_weights(data: pd.DataFrame) -> np.ndarray:
    """Give every curve equal total influence while retaining point-level rows."""
    if data.empty:
        raise ValueError("Cannot weight an empty dataset")
    counts = data["curve_id"].map(data["curve_id"].value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def study_balanced_weights(data: pd.DataFrame) -> np.ndarray:
    """Give every study equal influence and every curve equal influence within study."""
    if data.empty:
        raise ValueError("Cannot weight an empty dataset")
    curve_sizes = data["curve_id"].map(data["curve_id"].value_counts()).to_numpy(dtype=float)
    curves_per_study = data["study_id"].map(
        data.drop_duplicates("curve_id")["study_id"].value_counts()
    ).to_numpy(dtype=float)
    weights = 1.0 / (curve_sizes * curves_per_study)
    return weights / weights.mean()


def sample_weights(data: pd.DataFrame, weighting: str) -> np.ndarray:
    if weighting == "curve":
        return curve_balanced_weights(data)
    if weighting == "study":
        return study_balanced_weights(data)
    if weighting == "point":
        return np.ones(len(data), dtype=float)
    raise ValueError(f"Unknown weighting rule: {weighting}")


def model_catalog(predictor_set: str = "F-all") -> dict[str, ModelSpec]:
    all_numeric = tuple(NUMERIC_FEATURES)
    all_categorical = categorical_features_for(predictor_set)
    return {
        "mean": ModelSpec("mean", ("biodegradation_time_days",), (), ({},)),
        "ridge": ModelSpec(
            "ridge",
            all_numeric,
            all_categorical,
            tuple({"alpha": alpha} for alpha in (0.1, 1.0, 10.0, 100.0)),
        ),
        "random_forest": ModelSpec(
            "random_forest",
            all_numeric,
            all_categorical,
            (
                {"max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
                {"max_depth": 12, "min_samples_leaf": 1, "max_features": 1.0},
                {"max_depth": None, "min_samples_leaf": 3, "max_features": 0.7},
                {"max_depth": 16, "min_samples_leaf": 3, "max_features": 0.7},
                {"max_depth": None, "min_samples_leaf": 6, "max_features": 1.0},
                {"max_depth": 12, "min_samples_leaf": 6, "max_features": "sqrt"},
            ),
        ),
        "extra_trees": ModelSpec(
            "extra_trees",
            all_numeric,
            all_categorical,
            (
                {"max_depth": None, "min_samples_leaf": 1, "max_features": 1.0},
                {"max_depth": 12, "min_samples_leaf": 1, "max_features": 1.0},
                {"max_depth": None, "min_samples_leaf": 3, "max_features": 0.7},
                {"max_depth": 16, "min_samples_leaf": 3, "max_features": 0.7},
                {"max_depth": None, "min_samples_leaf": 6, "max_features": 1.0},
                {"max_depth": 12, "min_samples_leaf": 6, "max_features": "sqrt"},
            ),
        ),
        "xgboost": ModelSpec(
            "xgboost",
            all_numeric,
            all_categorical,
            (
                {"max_depth": 2, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
                {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 1},
                {"max_depth": 4, "learning_rate": 0.03, "subsample": 1.0, "colsample_bytree": 0.8, "min_child_weight": 1},
                {"max_depth": 2, "learning_rate": 0.06, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 3},
                {"max_depth": 3, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3},
                {"max_depth": 4, "learning_rate": 0.06, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 3},
            ),
        ),
    }


def make_estimator(name: str, parameters: dict[str, Any], random_state: int = 42):
    """Instantiate one deterministic candidate from the fixed model catalog."""
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(**parameters)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            **parameters,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            **parameters,
        )
    if name == "xgboost":
        return XGBRegressor(
            n_estimators=400,
            objective="reg:squarederror",
            tree_method="hist",
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            **parameters,
        )
    raise ValueError(f"Unknown model: {name}")
