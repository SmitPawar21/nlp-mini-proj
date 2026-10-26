"""
CLI Training Script — Train baseline or Transformer models for code-mixed NLP tasks.

Usage:
    python scripts/train.py --task sentiment --backbone google/muril-base-cased
    python scripts/train.py --task translation --backbone google/mt5-small
    python scripts/train.py --task ner --epochs 10 --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train code-mixed NLP models")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["translation", "sentiment", "ner", "qa", "intent"],
        help="NLP task to train",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Hugging Face backbone (e.g. google/muril-base-cased, google/mt5-small, xlm-roberta-base)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps",
    )

    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps

    logger.info(f"Starting training run for task: {args.task}")
    run_training(
        task=args.task,
        backbone=args.backbone,
        output_dir=args.output_dir,
        config_overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
