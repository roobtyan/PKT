"""Command line entry point for running training."""
from __future__ import annotations

import argparse
from pathlib import Path

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
