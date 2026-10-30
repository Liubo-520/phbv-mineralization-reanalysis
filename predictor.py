"""Inference and applicability warnings for the PHBV research-use model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


PREFERRED_CATEGORIES = {
    "biodegradation_condition": "aerobic",
    "degradation_mechanism": "microbial_biodegradation",
    "additives": "no",
    "degradation_environment": "soil",
    "additive_type_1": "not_applicable",
    "additive_type_2": "not_applicable",
    "additive_type_3": "not_applicable",
    "sample_shape_morphology": "films",
    "pha_degrading_microbes": "diverse_environmental",
}


class PHBVPredictor:
    """Load the frozen pipeline and return bounded estimates with warnings."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.bundle = joblib.load(self.model_path)
        # Avoid tiny run-to-run floating summation differences from parallel trees.
        if hasattr(self.bundle["estimator"], "n_jobs"):
            self.bundle["estimator"].n_jobs = 1
        self.numeric_features = list(self.bundle["numeric_features"])
        self.categorical_features = list(self.bundle["categorical_features"])
        self.features = self.numeric_features + self.categorical_features

    def default_record(self) -> dict[str, Any]:
        if "default_record" in self.bundle:
            return dict(self.bundle["default_record"])
        record: dict[str, Any] = {
            feature: float(schema["median"])
            for feature, schema in self.bundle["numeric_schema"].items()
        }
        for feature, levels in self.bundle["categorical_schema"].items():
            preferred = PREFERRED_CATEGORIES.get(feature)
            record[feature] = preferred if preferred in levels else levels[0]
        if record.get("additives") == "no":
            for index in (1, 2, 3):
                record[f"additive{index}_percentage_wt"] = 0.0
                record[f"additive_type_{index}"] = "not_applicable"
        return record

    @staticmethod
    def _validate_relationships(record: Mapping[str, Any]) -> None:
        percentages = [float(record.get(f"additive{index}_percentage_wt", 0.0) or 0.0) for index in (1, 2, 3)]
        types = [str(record.get(f"additive_type_{index}", "not_applicable")) for index in (1, 2, 3)]
        if str(record.get("additives", "no")) == "no" and any(value > 0 for value in percentages):
            raise ValueError("Additives are marked absent, but an additive percentage is nonzero.")
        for index, (value, additive_type) in enumerate(zip(percentages, types), start=1):
            if value > 0 and additive_type == "not_applicable":
                raise ValueError(
                    f"Additive {index} has a nonzero percentage but its type is not applicable."
                )

    def _frame_and_warnings(
        self, record: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, list[str]]:
        self._validate_relationships(record)
        defaults = self.default_record()
        normalized: dict[str, Any] = {}
        warnings: list[str] = []
        for feature in self.numeric_features:
            value = record.get(feature, defaults[feature])
            if value is None or value == "":
                normalized[feature] = np.nan
                warnings.append(f"{feature}: missing value will use the training median.")
                continue
            value = float(value)
            normalized[feature] = value
            schema = self.bundle["numeric_schema"][feature]
            if value < schema["minimum"] or value > schema["maximum"]:
                warnings.append(
                    f"{feature}: {value:g} is outside the training range "
                    f"[{schema['minimum']:g}, {schema['maximum']:g}]."
                )
        for feature in self.categorical_features:
            value = str(record.get(feature, defaults[feature]))
            normalized[feature] = value
            if value not in self.bundle["categorical_schema"][feature]:
                warnings.append(
                    f"{feature}: {value!r} is an unseen category and is outside the represented domain."
                )
        if warnings:
            warnings.append(
                "Applicability flags are warnings only; distance did not reliably calibrate study-level error."
            )
        else:
            warnings.append(
                "Inputs are within recorded marginal ranges; this does not guarantee that their combination or study is represented."
            )
        return pd.DataFrame([normalized], columns=self.features), warnings

    def _predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.bundle["preprocessor"].transform(frame[self.features])
        return np.clip(self.bundle["estimator"].predict(matrix), 0.0, 100.0)

    def _intervals(self, estimates: np.ndarray) -> tuple[np.ndarray, ...]:
        uncertainty = self.bundle["uncertainty"]
        represented = float(uncertainty["represented_study_curve_radius"])
        cross_study = float(uncertainty["cross_study_stress_radius"])
        return (
            np.clip(estimates - represented, 0.0, 100.0),
            np.clip(estimates + represented, 0.0, 100.0),
            np.clip(estimates - cross_study, 0.0, 100.0),
            np.clip(estimates + cross_study, 0.0, 100.0),
        )

    def _global_baseline(self, times: np.ndarray) -> np.ndarray | None:
        parameters = self.bundle.get("global_first_order_reference")
        if not parameters:
            return None
        asymptote = float(parameters["asymptote"])
        rate = float(parameters["rate_per_day"])
        return np.clip(asymptote * (1.0 - np.exp(-rate * times)), 0.0, 100.0)

    def predict_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        frame, warnings = self._frame_and_warnings(record)
        estimate = self._predict_frame(frame)
        rep_low, rep_high, cross_low, cross_high = self._intervals(estimate)
        result = {
            "estimate": float(estimate[0]),
            "represented_lower": float(rep_low[0]),
            "represented_upper": float(rep_high[0]),
            "cross_study_lower": float(cross_low[0]),
            "cross_study_upper": float(cross_high[0]),
            "warnings": warnings,
        }
        baseline = self._global_baseline(
            frame["biodegradation_time_days"].to_numpy(dtype=float)
        )
        if baseline is not None:
            result["global_baseline"] = float(baseline[0])
        return result

    def predict_curve(
        self,
        record: Mapping[str, Any],
        *,
        start_day: float,
        end_day: float,
        points: int = 100,
        monotone: bool = True,
    ) -> pd.DataFrame:
        if points < 2:
            raise ValueError("points must be at least two")
        if end_day <= start_day:
            raise ValueError("end_day must exceed start_day")
        curve_record = dict(record)
        curve_record["biodegradation_time_days"] = float(start_day)
        base, warnings = self._frame_and_warnings(curve_record)
        end_record = dict(curve_record)
        end_record["biodegradation_time_days"] = float(end_day)
        _, end_warnings = self._frame_and_warnings(end_record)
        warnings.extend(warning for warning in end_warnings if warning not in warnings)
        times = np.linspace(float(start_day), float(end_day), int(points))
        frame = pd.concat([base] * len(times), ignore_index=True)
        frame["biodegradation_time_days"] = times
        estimate = self._predict_frame(frame)
        if monotone:
            estimate = np.maximum.accumulate(estimate)
        rep_low, rep_high, cross_low, cross_high = self._intervals(estimate)
        output = pd.DataFrame(
            {
                "time_days": times,
                "estimate": estimate,
                "represented_lower": rep_low,
                "represented_upper": rep_high,
                "cross_study_lower": cross_low,
                "cross_study_upper": cross_high,
            }
        )
        baseline = self._global_baseline(times)
        if baseline is not None:
            output["global_baseline"] = baseline
        output.attrs["warnings"] = warnings
        return output
