from __future__ import annotations

import argparse
from pathlib import Path

from src.runner.experiment_runner import ExperimentRunner
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent leadership experiments.")
    parser.add_argument(
        "--config",
        default="src/config/pilot.json",
        help="Path to the experiment config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    runner = ExperimentRunner(config=config, root_dir=Path.cwd())
    runner.run()


if __name__ == "__main__":
    main()
