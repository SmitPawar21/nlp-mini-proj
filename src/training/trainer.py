"""
Unified training orchestrator.

Dispatches training to the appropriate model class based on task name,
loading config, datasets, and evaluation metrics automatically.

Usage:
    from src.training.trainer import run_training
    run_training(task="sentiment", backbone="google/muril-base-cased")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def _load_jsonl_as_dataset(path: Path) -> Any:
    """Load a JSONL file as a Hugging Face Dataset."""
    from datasets import Dataset

    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def run_training(
    task: str,
    backbone: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Run training for a specific task.

    Parameters
    ----------
    task : str
        One of: ``translation``, ``sentiment``, ``ner``, ``qa``, ``intent``.
    backbone : str, optional
        Override the model backbone from config.
    output_dir : str, optional
        Override the output directory.
    config_overrides : dict, optional
        Additional config overrides.
    """
    # Load merged config
    cfg = load_config(task, overrides=config_overrides)
    processed_dir = resolve_path(cfg, "processed_data_dir")

    # Determine task directory and load data
    task_dir = processed_dir / task
    train_path = task_dir / "train.jsonl"
    val_path = task_dir / "val.jsonl"

    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.error("Run data/scripts/download_datasets.py and data/scripts/preprocess.py first.")
        return

    logger.info(f"Loading training data from {task_dir}")
    train_ds = _load_jsonl_as_dataset(train_path)
    val_ds = _load_jsonl_as_dataset(val_path) if val_path.exists() else None

    logger.info(f"Train: {len(train_ds)} examples")
    if val_ds:
        logger.info(f"Val: {len(val_ds)} examples")

    # Resolve backbone and output dir
    if backbone is None:
        if task == "translation":
            backbone = cfg.get("model", {}).get("backbone", cfg["models"]["default_seq2seq"])
        else:
            backbone = cfg.get("model", {}).get("backbone", cfg["models"]["default_backbone"])

    if output_dir is None:
        model_dir = resolve_path(cfg, "model_dir")
        output_dir = str(model_dir / task)

    training_cfg = cfg.get("training", {})

    # Dispatch to task-specific model
    if task == "translation":
        from src.models.translation_model import TranslationModel

        model = TranslationModel(
            backbone=backbone,
            max_source_length=cfg.get("dataset", {}).get("max_source_length", 256),
            max_target_length=cfg.get("dataset", {}).get("max_target_length", 256),
        )
        model.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            num_epochs=training_cfg.get("num_epochs", 15),
            batch_size=training_cfg.get("batch_size", 8),
            learning_rate=training_cfg.get("learning_rate", 3e-5),
        )

    elif task == "sentiment":
        from src.models.sentiment_model import SentimentModel

        label_map = cfg.get("dataset", {}).get("label_map", {0: "negative", 1: "neutral", 2: "positive"})
        # Convert string keys to int if needed
        label_map = {int(k): v for k, v in label_map.items()}

        model = SentimentModel(
            backbone=backbone,
            num_labels=cfg.get("dataset", {}).get("num_labels", 3),
            label_map=label_map,
        )
        model.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            num_epochs=training_cfg.get("num_epochs", 10),
            batch_size=training_cfg.get("batch_size", 32),
            learning_rate=training_cfg.get("learning_rate", 2e-5),
            max_steps=training_cfg.get("max_steps", -1),
        )

    elif task == "ner":
        from src.models.ner_model import NERModel

        label_list = cfg.get("dataset", {}).get("label_list", None)
        model = NERModel(
            backbone=backbone,
            label_list=label_list,
        )
        model.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            num_epochs=training_cfg.get("num_epochs", 15),
            batch_size=training_cfg.get("batch_size", 16),
            learning_rate=training_cfg.get("learning_rate", 3e-5),
        )

    elif task == "qa":
        from src.models.qa_model import QAModel

        model = QAModel(
            backbone=backbone,
            max_length=cfg.get("dataset", {}).get("max_context_length", 384),
            doc_stride=cfg.get("dataset", {}).get("doc_stride", 128),
        )
        model.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            num_epochs=training_cfg.get("num_epochs", 10),
            batch_size=training_cfg.get("batch_size", 16),
            learning_rate=training_cfg.get("learning_rate", 3e-5),
        )

    elif task == "intent":
        from src.models.intent_model import IntentModel

        model = IntentModel(
            backbone=backbone,
            num_labels=cfg.get("dataset", {}).get("num_labels", 26),
        )
        model.train(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            num_epochs=training_cfg.get("num_epochs", 15),
            batch_size=training_cfg.get("batch_size", 32),
            learning_rate=training_cfg.get("learning_rate", 2e-5),
            max_steps=training_cfg.get("max_steps", -1),
        )

    else:
        logger.error(f"Unknown task: {task}")
        return

    logger.info(f"Training complete for task: {task}")
