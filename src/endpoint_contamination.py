"""Controlled endpoint-contamination training-set construction and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import hierarchical_bootstrap_interval, regression_metrics
from src.models import ModelSpec, make_estimator, sample_weights
from src.preprocessing import build_preprocessor
from src.reviewer_analyses import paired_difference_diagnostics


def resample_complete_curves_to_size(
    data: pd.DataFrame,
    *,
    target_rows: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample complete curves with replacement until at least target_rows are present."""
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    if data.empty:
        raise ValueError("Cannot resample an empty dataset")
    curve_keys = list(data[["study_id", "curve_id"]].drop_duplicates().itertuples(index=False, name=None))
    rng = np.random.default_rng(random_state)
    parts: list[pd.DataFrame] = []
    row_count = 0
    draw = 0
    while row_count < target_rows:
        study_id, curve_id = curve_keys[int(rng.integers(len(curve_keys)))]
        part = data.loc[
            data["study_id"].eq(study_id) & data["curve_id"].eq(curve_id)
        ].copy()
        part["bootstrap_curve_id"] = f"draw_{draw:04d}_{study_id}_{curve_id}"
        parts.append(part)
        row_count += len(part)
        draw += 1
    return pd.concat(parts, ignore_index=True)


def endpoint_training_sets(
    co2_data: pd.DataFrame,
    mass_loss_data: pd.DataFrame,
    *,
    held_out_study: int,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Construct disjoint CO2-only, contaminated, and size-control training sets."""
    co2_train = co2_data.loc[~co2_data["study_id"].eq(held_out_study)].copy()
    if co2_train.empty:
        raise ValueError("The held-out study leaves no CO2 training data")
    mass_train = mass_loss_data.loc[~mass_loss_data["study_id"].eq(held_out_study)].copy()
    mixed = pd.concat([co2_train, mass_train], ignore_index=True, sort=False)
    size_matched = resample_complete_curves_to_size(
        co2_train,
        target_rows=len(mixed),
        random_state=random_state,
    )
    return {
        "co2_only": co2_train,
        "co2_plus_mass_loss": mixed.copy(),
        "co2_size_matched": size_matched,
        "co2_plus_mass_loss_with_endpoint_label": mixed.copy(),
    }


def fit_endpoint_training_condition(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    spec: ModelSpec,
    parameters: dict,
    weighting: str,
    random_state: int,
) -> pd.DataFrame:
    """Fit one controlled training condition and predict the unchanged test rows."""
    features = list(spec.numeric_features + spec.categorical_features)
    preprocessor = build_preprocessor(spec.numeric_features, spec.categorical_features)
    x_train = preprocessor.fit_transform(train[features])
    x_test = preprocessor.transform(test[features])
    estimator = make_estimator(spec.name, parameters, random_state=random_state)
    estimator.fit(
        x_train,
        train["biodegradation_percentage"].to_numpy(dtype=float),
        sample_weight=sample_weights(train, weighting),
    )
    prediction = estimator.predict(x_test)
    return pd.DataFrame(
        {
            "study_id": test["study_id"].to_numpy(),
            "curve_id": test["curve_id"].to_numpy(),
            "y_true": test["biodegradation_percentage"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )


def endpoint_contamination_predictions(
    co2_data: pd.DataFrame,
    mass_loss_data: pd.DataFrame,
    *,
    spec: ModelSpec,
    parameters: dict,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate four training-set conditions against identical held-out CO2 studies."""
    prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for fold_index, held_out_study in enumerate(sorted(co2_data["study_id"].unique())):
        test = co2_data.loc[co2_data["study_id"].eq(held_out_study)].copy()
        sets = endpoint_training_sets(
            co2_data,
            mass_loss_data,
            held_out_study=held_out_study,
            random_state=random_state + fold_index,
        )
        for condition_index, (condition, train_source) in enumerate(sets.items()):
            train = train_source.copy()
            if condition == "co2_size_matched":
                train["source_curve_id"] = train["curve_id"]
                train["curve_id"] = train["bootstrap_curve_id"]
            condition_spec = spec
            if condition == "co2_plus_mass_loss_with_endpoint_label":
                condition_spec = ModelSpec(
                    name=spec.name,
                    numeric_features=spec.numeric_features,
                    categorical_features=spec.categorical_features + ("endpoint_class",),
                    candidates=spec.candidates,
                )
            fold_predictions = fit_endpoint_training_condition(
                train,
                test,
                spec=condition_spec,
                parameters=parameters,
                weighting="study",
                random_state=random_state + fold_index * 10 + condition_index,
            )
            fold_predictions.insert(0, "test_row_index", test.index.to_numpy())
            fold_predictions.insert(0, "outer_fold", fold_index)
            fold_predictions.insert(0, "condition", condition)
            fold_predictions.insert(0, "task", "endpoint_contamination_loso")
            prediction_parts.append(fold_predictions)
            audit_rows.append(
                {
                    "condition": condition,
                    "outer_fold": fold_index,
                    "held_out_study": held_out_study,
                    "training_rows": int(len(train)),
                    "training_curves": int(train["curve_id"].nunique()),
                    "training_studies": int(train["study_id"].nunique()),
                    "mass_loss_training_rows": int(
                        train["endpoint_class"].eq("mass_loss").sum()
                    ),
                    "held_out_rows": int(len(test)),
                    "held_out_study_in_training": bool(
                        held_out_study in set(train["study_id"])
                    ),
                    "endpoint_label_feature": bool(
                        condition == "co2_plus_mass_loss_with_endpoint_label"
                    ),
                }
            )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(audit_rows)


def random_row_contamination_predictions(
    co2_data: pd.DataFrame,
    mass_loss_data: pd.DataFrame,
    *,
    spec: ModelSpec,
    parameters: dict,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide a declared within-trajectory random-row contamination reference."""
    indices = np.arange(len(co2_data))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=random_state,
        shuffle=True,
    )
    co2_train = co2_data.iloc[train_index].copy()
    test = co2_data.iloc[test_index].copy()
    mixed = pd.concat([co2_train, mass_loss_data], ignore_index=True, sort=False)
    conditions = {
        "co2_only": co2_train,
        "co2_plus_mass_loss": mixed.copy(),
        "co2_size_matched": resample_complete_curves_to_size(
            co2_train,
            target_rows=len(mixed),
            random_state=random_state,
        ),
        "co2_plus_mass_loss_with_endpoint_label": mixed.copy(),
    }
    prediction_parts = []
    audit_rows = []
    for condition_index, (condition, train_source) in enumerate(conditions.items()):
        train = train_source.copy()
        if condition == "co2_size_matched":
            train["source_curve_id"] = train["curve_id"]
            train["curve_id"] = train["bootstrap_curve_id"]
        condition_spec = spec
        if condition == "co2_plus_mass_loss_with_endpoint_label":
            condition_spec = ModelSpec(
                name=spec.name,
                numeric_features=spec.numeric_features,
                categorical_features=spec.categorical_features + ("endpoint_class",),
                candidates=spec.candidates,
            )
        fold_predictions = fit_endpoint_training_condition(
            train,
            test,
            spec=condition_spec,
            parameters=parameters,
            weighting="curve",
            random_state=random_state + condition_index,
        )
        fold_predictions.insert(0, "test_row_index", test.index.to_numpy())
        fold_predictions.insert(0, "condition", condition)
        fold_predictions.insert(0, "task", "endpoint_contamination_random_row_reference")
        prediction_parts.append(fold_predictions)
        train_source_curves = set(co2_train["curve_id"])
        test_curves = set(test["curve_id"])
        audit_rows.append(
            {
                "condition": condition,
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_test_curve_overlap": int(len(train_source_curves & test_curves)),
                "mass_loss_training_rows": int(
                    train["endpoint_class"].eq("mass_loss").sum()
                ),
                "endpoint_label_feature": bool(
                    condition == "co2_plus_mass_loss_with_endpoint_label"
                ),
            }
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(audit_rows)


def summarize_endpoint_contamination(
    predictions: pd.DataFrame,
    *,
    reference_condition: str = "co2_only",
    bootstrap_iterations: int = 5000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize condition errors and paired study-level differences."""
    metric_rows = []
    difference_rows = []
    for group_index, ((task, condition), part) in enumerate(
        predictions.groupby(["task", "condition"], sort=False)
    ):
        interval = hierarchical_bootstrap_interval(
            part,
            metric="study_curve_macro_mae",
            iterations=bootstrap_iterations,
            random_state=random_state + group_index,
        )
        metric_rows.append(
            {
                "task": task,
                "condition": condition,
                **regression_metrics(part),
                "study_clustered_ci_lower": interval["lower"],
                "study_clustered_ci_upper": interval["upper"],
            }
        )

    for task, part in predictions.groupby("task", sort=False):
        losses = part.assign(
            absolute_error=lambda frame: np.abs(frame["y_true"] - frame["y_pred"])
        )
        curve_losses = (
            losses.groupby(["condition", "study_id", "curve_id"], sort=False)[
                "absolute_error"
            ]
            .mean()
            .reset_index()
        )
        study_losses = (
            curve_losses.groupby(["condition", "study_id"], sort=False)["absolute_error"]
            .mean()
            .unstack("condition")
        )
        if reference_condition not in study_losses:
            raise ValueError(f"Missing reference condition: {reference_condition}")
        for condition in study_losses.columns:
            if condition == reference_condition:
                continue
            paired = study_losses[[condition, reference_condition]].dropna()
            differences = paired[condition] - paired[reference_condition]
            difference_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "reference_condition": reference_condition,
                    **paired_difference_diagnostics(differences.to_numpy()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(difference_rows)
