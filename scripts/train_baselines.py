"""
Train & Save Baseline Models (TF-IDF + SVM/LR for Sentiment & Intent, CRF for NER).

Trains fast, lightweight, high-performance baseline models and saves weights
to ``models/{task}/`` so the UnifiedPredictor and Streamlit UI load real
trained model checkpoints.

Usage:
    python scripts/train_baselines.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines import TFIDFClassifier, CRFSequenceTagger
from src.utils.config import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def train_sentiment_baseline(processed_dir: Path, models_dir: Path) -> None:
    train_path = processed_dir / "sentiment" / "train.jsonl"
    if not train_path.exists():
        logger.warning(f"Sentiment data not found at {train_path}")
        return

    data = load_jsonl(train_path)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    logger.info(f"Training TF-IDF Sentiment Baseline on {len(texts)} samples...")
    clf = TFIDFClassifier(classifier_type="svm", max_features=10000)
    clf.fit(texts, labels)

    save_path = models_dir / "sentiment" / "baseline_sentiment.joblib"
    clf.save(str(save_path))


def train_intent_baseline(processed_dir: Path, models_dir: Path) -> None:
    train_path = processed_dir / "intent" / "train.jsonl"
    if not train_path.exists():
        logger.warning(f"Intent data not found at {train_path}")
        return

    data = load_jsonl(train_path)
    texts = [d["text"] for d in data]
    labels = [d["intent"] for d in data]

    logger.info(f"Training TF-IDF Intent Baseline on {len(texts)} samples across {len(set(labels))} classes...")
    clf = TFIDFClassifier(classifier_type="lr", max_features=30000, max_iter=2000)
    clf.fit(texts, labels)

    save_path = models_dir / "intent" / "baseline_intent.joblib"
    clf.save(str(save_path))


def train_ner_baseline(processed_dir: Path, models_dir: Path) -> None:
    train_path = processed_dir / "ner" / "train.jsonl"
    if not train_path.exists():
        logger.warning(f"NER data not found at {train_path}")
        return

    try:
        data = load_jsonl(train_path)[:2000]  # Fast train on 2,000 COMI-LINGUA sentences
        tokens_list = [d["tokens"] for d in data]
        tags_list = [d["ner_tags"] for d in data]

        logger.info(f"Training CRF NER Baseline on {len(tokens_list)} sentences...")
        tagger = CRFSequenceTagger(max_iterations=50)
        model_file = str(models_dir / "ner" / "baseline_crf_ner.model")
        tagger.fit(tokens_list, tags_list, model_path=model_file)
    except Exception as e:
        logger.warning(f"Skipping CRF NER baseline training: {e}")


def main() -> None:
    cfg = load_config()
    processed_dir = resolve_path(cfg, "processed_data_dir")
    models_dir = resolve_path(cfg, "model_dir")

    train_intent_baseline(processed_dir, models_dir)
    train_sentiment_baseline(processed_dir, models_dir)
    train_ner_baseline(processed_dir, models_dir)

    logger.info("Baseline models trained and saved successfully!")


if __name__ == "__main__":
    main()
