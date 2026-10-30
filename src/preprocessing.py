"""Fold-local preprocessing for mixed PHBV descriptors."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "biodegradation_time_days",
    "adjusted_hb_ratio_formulation_mol",
    "adjusted_hv_ratio_formulation_mol",
    "t_biodeg",
    "additive1_percentage_wt",
    "additive2_percentage_wt",
    "additive3_percentage_wt",
]

CATEGORICAL_FEATURES = [
    "biodegradation_condition",
    "degradation_mechanism",
    "additives",
    "degradation_environment",
    "additive_type_1",
    "additive_type_2",
    "additive_type_3",
    "sample_shape_morphology",
    "pha_degrading_microbes",
]


def model_matrix(data):
    """Select the prespecified model inputs in a fixed order."""
    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Create an unfitted transformer; callers fit it on training folds only."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(numeric_features)), ("categorical", categorical, list(categorical_features))],
        sparse_threshold=1.0,
    )
