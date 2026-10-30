"""Hierarchy-preserving grouped permutation importance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.models import curve_balanced_weights, make_estimator, model_catalog
from src.preprocessing import build_preprocessor
from src.sensitivity import FROZEN_RF_PARAMETERS


FEATURE_GROUPS = {
    "time": ["biodegradation_time_days"],
    "copolymer_composition": ["adjusted_hb_ratio_formulation_mol", "adjusted_hv_ratio_formulation_mol"],
    "thermal_condition": ["t_biodeg"],
    "biodegradation_context": [
        "biodegradation_condition",
        "degradation_mechanism",
        "degradation_environment",
        "pha_degrading_microbes",
    ],
    "additive_package": [
        "additives",
        "additive_type_1",
        "additive1_percentage_wt",
        "additive_type_2",
        "additive2_percentage_wt",
        "additive_type_3",
        "additive3_percentage_wt",
    ],
    "sample_morphology": ["sample_shape_morphology"],
}


def permute_feature_group(data: pd.DataFrame, features: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Exchange a joint static feature block among complete curves."""
    output = data.copy()
    curve_values = output.drop_duplicates("curve_id")[["curve_id"] + features].reset_index(drop=True)
    shuffled = curve_values[features].iloc[rng.permutation(len(curve_values))].reset_index(drop=True)
    for feature in features:
        mapping = dict(zip(curve_values["curve_id"], shuffled[feature]))
        output[feature] = output["curve_id"].map(mapping)
    return output


def _permute_time_within_curves(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    output = data.copy()
    for _, indices in output.groupby("curve_id", sort=False).groups.items():
        output.loc[indices, "biodegradation_time_days"] = rng.permutation(
            output.loc[indices, "biodegradation_time_days"].to_numpy()
        )
    return output


def run_grouped_permutation_importance(
    root: Path, repeats: int = 5, random_state: int = 42, predictor_set: str = "F-all"
) -> dict[str, pd.DataFrame]:
    data = pd.read_csv(root / "data/processed/D3_endpoint_audited_mineralization.csv")
    predictions = pd.read_csv(root / "results/predictions/strict_oof_predictions.csv")
    reference = predictions[predictions["model"].eq("random_forest")]
    spec = model_catalog(predictor_set)["random_forest"]
    features = list(spec.numeric_features + spec.categorical_features)
    rows = []

    for task, task_predictions in reference.groupby("task", sort=False):
        for outer_fold, fold_predictions in task_predictions.groupby("outer_fold", sort=False):
            test_idx = np.sort(fold_predictions["row_index"].to_numpy(dtype=int))
            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            preprocessor = build_preprocessor(spec.numeric_features, spec.categorical_features)
            x_train = preprocessor.fit_transform(train[features])
            x_test = preprocessor.transform(test[features])
            estimator = make_estimator("random_forest", FROZEN_RF_PARAMETERS, random_state + outer_fold)
            estimator.fit(
                x_train,
                train["biodegradation_percentage"].to_numpy(dtype=float),
                sample_weight=curve_balanced_weights(train),
            )
            base_frame = pd.DataFrame(
                {
                    "study_id": test["study_id"].to_numpy(),
                    "curve_id": test["curve_id"].to_numpy(),
                    "y_true": test["biodegradation_percentage"].to_numpy(dtype=float),
                    "y_pred": estimator.predict(x_test),
                }
            )
            baseline = regression_metrics(base_frame)["curve_macro_mae"]
            for repeat in range(repeats):
                for group_index, (group, group_features) in enumerate(FEATURE_GROUPS.items()):
                    rng = np.random.default_rng(random_state + outer_fold * 1000 + repeat * 100 + group_index)
                    permuted = (
                        _permute_time_within_curves(test, rng)
                        if group == "time"
                        else permute_feature_group(test, group_features, rng)
                    )
                    prediction = estimator.predict(preprocessor.transform(permuted[features]))
                    permuted_frame = base_frame.copy()
                    permuted_frame["y_pred"] = prediction
                    permuted_mae = regression_metrics(permuted_frame)["curve_macro_mae"]
                    rows.append(
                        {
                            "task": task,
                            "outer_fold": outer_fold,
                            "repeat": repeat,
                            "feature_group": group,
                            "baseline_curve_macro_mae": baseline,
                            "permuted_curve_macro_mae": permuted_mae,
                            "mae_increase": permuted_mae - baseline,
                        }
                    )
    importance = pd.DataFrame(rows)
    summary = (
        importance.groupby(["task", "feature_group"])
        .agg(
            mean_mae_increase=("mae_increase", "mean"),
            median_mae_increase=("mae_increase", "median"),
            sd_mae_increase=("mae_increase", "std"),
            lower_2_5=("mae_increase", lambda x: x.quantile(0.025)),
            upper_97_5=("mae_increase", lambda x: x.quantile(0.975)),
            evaluations=("mae_increase", "size"),
        )
        .reset_index()
    )
    (root / "results/interpretability").mkdir(parents=True, exist_ok=True)
    (root / "results/metrics").mkdir(parents=True, exist_ok=True)
    importance.to_csv(root / "results/interpretability/grouped_permutation_importance.csv", index=False)
    summary.to_csv(root / "results/metrics/grouped_permutation_importance_summary.csv", index=False)
    return {"importance": importance, "summary": summary}


if __name__ == "__main__":
    run_grouped_permutation_importance(Path(__file__).resolve().parents[1])
