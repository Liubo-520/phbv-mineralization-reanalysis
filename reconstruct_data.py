"""Reconstruct D0-D3 from the versioned public workbook and published CSV."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from src.reconstruct_dataset import reconstruct


EXPECTED = {
    "workbook": "eb055d58082c9715816a8e8abd2bb570c52c4062863fdba49b969564c94b4c9e",
    "published_csv": "418ded950718c40f54a8304a46dd56e7626753ac3bb9808c82091d14b99e5297",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--published-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/reconstructed"))
    args = parser.parse_args()
    for label in EXPECTED:
        path = getattr(args, label)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != EXPECTED[label]:
            raise ValueError(f"{label}: input hash does not match the documented public record.")
    layers = reconstruct(args.workbook, args.published_csv,
                         args.output / "processed", args.output / "audit")
    d3 = layers["D3"]
    print(f"D3: {len(d3)} observations, {d3.curve_id.nunique()} curves, "
          f"{d3.study_id.nunique()} studies.")


if __name__ == "__main__":
    main()
