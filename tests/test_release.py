"""Release plumbing checks; synthetic grouping rows are not PHBV measurements."""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import PHBVPredictor
from src.splitting import curve_group_folds, leave_one_study_out_folds


def test_manifest_and_frozen_model_identity():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    predictor = PHBVPredictor(ROOT / "models/phbv_rf_bundle.joblib")
    assert not set(predictor.features) & {
        "study_id", "instance", "sample_name", "curve_id", "curve_family_id",
        "degradation_mechanism", "pha_degrading_microbes",
    }
    assert len(predictor.features) == 14


def test_frozen_prediction_and_applicability_flags():
    predictor = PHBVPredictor(ROOT / "models/phbv_rf_bundle.joblib")
    record = predictor.default_record()
    first = predictor.predict_record(record)
    assert first == predictor.predict_record(record)
    assert 0 <= first["estimate"] <= 100
    assert first["cross_study_lower"] <= first["represented_lower"] <= first["represented_upper"] <= first["cross_study_upper"]
    record["biodegradation_time_days"] = 999
    assert any("outside" in warning for warning in predictor.predict_record(record)["warnings"])


def test_synthetic_fold_boundaries_keep_repeated_units_together():
    data = pd.DataFrame({
        "study_id": np.repeat([1, 2, 3], 8),
        "curve_id": np.repeat(np.arange(6), 4),
        "curve_family_id": np.repeat(np.arange(6), 4),
    })
    for train, test in curve_group_folds(data, n_splits=3, group_column="curve_family_id"):
        assert not set(data.iloc[train].curve_id) & set(data.iloc[test].curve_id)
    for train, test in leave_one_study_out_folds(data):
        assert not set(data.iloc[train].study_id) & set(data.iloc[test].study_id)
