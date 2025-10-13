"""Command line entry point for running model inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ensure repository root is importable when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pkt import Trainer, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to run inference on (default: val).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path to load before running inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for inference (e.g., cpu, cuda:0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save predictions via torch.save.",
    )
    parser.add_argument(
        "--return-logits",
        action="store_true",
        help="Include raw logits in the saved/returned results.",
    )
    parser.add_argument(
        "--return-targets",
        action="store_true",
        help="Include ground-truth targets in the saved/returned results (requires dataset with labels).",
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
    results = trainer.predict(
        split=args.split,
        return_logits=args.return_logits,
        return_targets=args.return_targets,
    )

    num_preds = results["preds"].shape[0]
    if args.output is not None:
        torch.save(results, args.output)
        print(f"Saved {num_preds} predictions to {args.output}")
    else:
        preview = results["preds"][:5].tolist()
        print(f"Generated {num_preds} predictions. Preview of first 5: {preview}")


if __name__ == "__main__":
    main()
