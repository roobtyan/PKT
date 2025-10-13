"""Command line entry point for running training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the repository root is importable when executing the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pkt import Trainer, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training using a YAML configuration file")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
