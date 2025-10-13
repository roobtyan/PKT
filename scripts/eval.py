"""Command line entry point for evaluating a trained model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is importable when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pkt import Trainer, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a checkpoint to load before evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for evaluation (e.g., cpu, cuda:0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if args.device:
        cfg.trainer.device = args.device
    if args.checkpoint is not None:
        cfg.trainer.resume_from = str(args.checkpoint)

    trainer = Trainer(cfg)
    metrics = trainer.evaluate(split=args.split)

    print(f"Evaluation results on split '{args.split}':")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
