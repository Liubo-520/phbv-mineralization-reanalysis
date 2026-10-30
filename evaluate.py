"""Run the primary F-pre nested curve-family and study-disjoint evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments import run_strict_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--models", nargs="+",
                        choices=("mean", "ridge", "random_forest", "extra_trees", "xgboost"),
                        default=("mean", "ridge", "random_forest", "extra_trees", "xgboost"))
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be positive.")
    run_strict_experiments(
        args.data, args.output, model_names=args.models,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=42, predictor_set="F-pre",
    )


if __name__ == "__main__":
    main()
