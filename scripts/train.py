"""Command line entry point for running training or pipelines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Make sure the repository root is importable when executing the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pkt import Trainer, load_yaml_config
from pkt.engine.runners import build_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training using a YAML configuration file")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    return parser.parse_args()


def _load_raw_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    raw_cfg = _load_raw_config(args.config)
    if isinstance(raw_cfg, dict) and "runner" in raw_cfg:
        runner = build_runner(raw_cfg)
        runner.run()
    else:
        cfg = load_yaml_config(args.config)
        trainer = Trainer(cfg)
        trainer.train()


if __name__ == "__main__":
    main()
