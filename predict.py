"""Explicit command-line inference with the frozen PHBV research model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictor import PHBVPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", action="store_true")
    source.add_argument("--input", type=Path, help="JSON object of experiment-time conditions")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models/phbv_rf_bundle.joblib")
    args = parser.parse_args()
    predictor = PHBVPredictor(args.model)
    record = predictor.default_record() if args.example else json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        parser.error("Input JSON must be an object.")
    print(json.dumps(predictor.predict_record(record), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
