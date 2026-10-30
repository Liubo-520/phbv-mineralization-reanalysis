"""Train and serialize the final research-use RF bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import least_squares

from src.models import make_estimator, model_catalog, study_balanced_weights
from src.preprocessing import build_preprocessor
from src.reviewer_analyses import predictor_feature_sets
from src.sensitivity import FROZEN_RF_PARAMETERS
from src.uncertainty import curve_maximum_scores, finite_sample_quantile


def deployment_model_spec():
    """Return the conservative feature schema that is knowable before testing."""
    return predictor_feature_sets()["F-pre"]


def empirical_curve_radius(predictions: pd.DataFrame, coverage: float = 0.90) -> float:
    return finite_sample_quantile(curve_maximum_scores(predictions), coverage=coverage)


def empirical_study_radius(predictions: pd.DataFrame, coverage: float = 0.90) -> float:
    errors = (predictions["y_true"] - predictions["y_pred"]).abs()
    study_scores = errors.groupby(predictions["study_id"], sort=False).max()
    return finite_sample_quantile(study_scores, coverage=coverage)


def bounded_interval(center: float, radius: float) -> tuple[float, float]:
    return max(0.0, float(center) - float(radius)), min(100.0, float(center) + float(radius))


def fit_global_first_order(data: pd.DataFrame) -> dict[str, float]:
    """Fit the study-balanced global kinetic reference used by the public artifact."""
    time = data["biodegradation_time_days"].to_numpy(dtype=float)
    response = data["biodegradation_percentage"].to_numpy(dtype=float)
    weights = study_balanced_weights(data)

    def residual(parameters: np.ndarray) -> np.ndarray:
        asymptote, rate = parameters
        prediction = asymptote * (1.0 - np.exp(-rate * time))
        return np.sqrt(weights) * (prediction - response)

    fit = least_squares(
        residual,
        x0=np.array([90.0, 0.02]),
        bounds=(np.array([1e-6, 1e-8]), np.array([100.0, 10.0])),
        max_nfev=20_000,
    )
    return {"asymptote": float(fit.x[0]), "rate_per_day": float(fit.x[1])}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_final_model(root: Path, coverage: float = 0.90) -> dict:
    data_path = root / "data/processed/D3_endpoint_audited_mineralization.csv"
    data = pd.read_csv(data_path)
    spec = deployment_model_spec()
    features = list(spec.numeric_features + spec.categorical_features)
    preprocessor = build_preprocessor(spec.numeric_features, spec.categorical_features)
    transformed = preprocessor.fit_transform(data[features])
    estimator = make_estimator("random_forest", FROZEN_RF_PARAMETERS, random_state=42)
    estimator.fit(
        transformed,
        data["biodegradation_percentage"].to_numpy(dtype=float),
        sample_weight=study_balanced_weights(data),
    )

    fixed_oof = pd.read_csv(root / "results/predictions/fpre_deployment_predictions.csv")
    represented = fixed_oof.loc[
        fixed_oof["task"].eq("task_b_curve_grouped_nested_cv")
    ]
    stress = fixed_oof.loc[
        fixed_oof["task"].eq("task_c_leave_one_study_out")
    ]
    if represented.empty or stress.empty:
        raise ValueError("Exact frozen-configuration OOF predictions are missing")
    radii = {
        "coverage": coverage,
        "represented_study_curve_radius": empirical_curve_radius(represented, coverage),
        "cross_study_stress_radius": empirical_study_radius(stress, coverage),
        "represented_unit": "maximum absolute residual per held-out curve",
        "cross_study_unit": "maximum absolute residual over all curves and times in each held-out study",
        "interpretation": (
            "Finite-sample empirical residual radii for the exact frozen configuration; "
            "the cross-study value is a descriptive stress result, not a prospective "
            "coverage guarantee."
        ),
    }
    numeric_schema = {
        feature: {
            "minimum": None if pd.isna(data[feature].min()) else float(data[feature].min()),
            "median": None if pd.isna(data[feature].median()) else float(data[feature].median()),
            "maximum": None if pd.isna(data[feature].max()) else float(data[feature].max()),
        }
        for feature in spec.numeric_features
    }
    categorical_schema = {
        feature: sorted(map(str, data[feature].dropna().unique()))
        for feature in spec.categorical_features
    }
    complete = data.dropna(subset=features)
    default_source = complete.iloc[0] if not complete.empty else data.iloc[0]
    default_record = {
        feature: (float(default_source[feature]) if feature in spec.numeric_features else str(default_source[feature]))
        for feature in features
    }
    metrics = pd.read_csv(root / "results/metrics/predictor_availability_metrics.csv")
    strict_rf = metrics.loc[metrics["model"].eq("random_forest_f_pre")]
    validation_summary = {
        row["task"]: {
            key: value
            for key, value in row.items()
            if key
            in {
                "primary_mae_metric",
                "primary_mae",
                "primary_mae_ci_lower",
                "primary_mae_ci_upper",
                "point_mae",
                "point_rmse",
                "point_r2",
                "curve_macro_mae",
                "study_curve_macro_mae",
            }
        }
        for row in strict_rf.to_dict(orient="records")
    }
    global_baseline_metrics = pd.read_csv(
        root / "results/metrics/global_kinetic_baseline_metrics.csv"
    )
    global_baseline_summary = {
        row["task"]: {
            key: value
            for key, value in row.items()
            if key
            in {
                "primary_mae_metric",
                "primary_mae",
                "primary_mae_ci_lower",
                "primary_mae_ci_upper",
                "point_mae",
                "point_r2",
                "curve_macro_mae",
                "study_curve_macro_mae",
            }
        }
        for row in global_baseline_metrics.to_dict(orient="records")
    }
    fixed_metrics = pd.read_csv(root / "results/metrics/fpre_deployment_metrics.csv")
    exact_fixed_metrics = fixed_metrics.copy()
    exact_configuration_summary = {
        row["task"]: {
            key: value
            for key, value in row.items()
            if key
            in {
                "primary_mae_metric",
                "primary_mae",
                "primary_mae_ci_lower",
                "primary_mae_ci_upper",
                "point_mae",
                "point_r2",
                "curve_macro_mae",
                "study_curve_macro_mae",
            }
        }
        for row in exact_fixed_metrics.to_dict(orient="records")
    }
    bundle = {
        "preprocessor": preprocessor,
        "estimator": estimator,
        "numeric_features": list(spec.numeric_features),
        "categorical_features": list(spec.categorical_features),
        "numeric_schema": numeric_schema,
        "categorical_schema": categorical_schema,
        "default_record": default_record,
        "parameters": FROZEN_RF_PARAMETERS,
        "uncertainty": radii,
        "training_rows": len(data),
        "training_curves": int(data["curve_id"].nunique()),
        "training_studies": int(data["study_id"].nunique()),
        "training_curve_families": int(data["curve_family_id"].nunique()),
        "training_data_version": "D3_endpoint_audited_mineralization",
        "predictor_set": "F-pre",
        "predictor_availability_rule": (
            "Only variables expected to be knowable before a biodegradation experiment; "
            "source-curated degradation mechanism and PHA-degrading-microbe labels are excluded."
        ),
        "training_weighting": "equal study weight, then equal curve weight within study",
        "configuration_rule": (
            "Modal random-forest hyperparameter set across the 11 correctly nested "
            "Task C outer selections; exact fixed-configuration OOF sensitivity is retained."
        ),
        "validation_summary": validation_summary,
        "exact_configuration_validation_summary": exact_configuration_summary,
        "global_first_order_reference": fit_global_first_order(data),
        "global_baseline_validation_summary": global_baseline_summary,
        "model_version": "3.0.0",
        "data_sha256": _sha256(data_path),
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "phbv_rf_bundle.joblib", compress=3)
    metadata = {
        key: value for key, value in bundle.items()
        if key not in {"preprocessor", "estimator"}
    }
    metadata["strict_random_forest_performance"] = strict_rf.to_dict(orient="records")
    metadata["exact_frozen_configuration_performance"] = exact_fixed_metrics.to_dict(
        orient="records"
    )
    (model_dir / "model_card_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return bundle


if __name__ == "__main__":
    train_final_model(Path(__file__).resolve().parents[1])
