"""
Unified preprocessing pipeline — transforms raw/local datasets into clean,
task-ready train/val/test splits saved to ``data/processed/``.

Supports local COMI-LINGUA CSV files in ``data/COMI-LINGUA/`` as well as
downloaded Hugging Face datasets.

Tasks preprocessed:
1. Translation (MT_train.csv / MT_test.csv)
2. NER (NER_train.csv / NER_test.csv)
3. Sentiment Analysis (L3Cube-HingSent or local sentiment data)
4. Question Answering (IndicQA Hindi or local QA data)
5. Intent Detection (Synthetic Hinglish ATIS/SNIPS)

Usage:
    python data/scripts/preprocess.py
    python data/scripts/preprocess.py --tasks translation ner
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.augmentation import SyntheticCodeMixer
from src.preprocessing.normalizer import CodeMixedNormalizer
from src.utils.config import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def _save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dicts as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in data:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(data)} examples → {path}")


def _deduplicate(
    data: List[Dict[str, Any]],
    text_key: str,
) -> List[Dict[str, Any]]:
    """Remove duplicate entries based on text hash."""
    seen: Set[str] = set()
    unique = []
    for item in data:
        h = _text_hash(str(item.get(text_key, "")))
        if h not in seen:
            seen.add(h)
            unique.append(item)
    removed = len(data) - len(unique)
    if removed > 0:
        logger.info(f"  Removed {removed} duplicates")
    return unique


def _split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train and validation splits."""
    import random

    rng = random.Random(seed)
    data_copy = list(data)
    rng.shuffle(data_copy)

    n = len(data_copy)
    train_end = int(n * train_ratio)

    return data_copy[:train_end], data_copy[train_end:]


def _check_leakage(
    train: List[Dict], val: List[Dict], test: List[Dict], text_key: str
) -> None:
    """Verify no text overlap between splits."""
    train_hashes = {_text_hash(str(d.get(text_key, ""))) for d in train}
    val_hashes = {_text_hash(str(d.get(text_key, ""))) for d in val}
    test_hashes = {_text_hash(str(d.get(text_key, ""))) for d in test}

    tv = train_hashes & val_hashes
    tt = train_hashes & test_hashes
    vt = val_hashes & test_hashes

    if tv or tt or vt:
        logger.warning(
            f"  DATA LEAKAGE DETECTED: train∩val={len(tv)}, "
            f"train∩test={len(tt)}, val∩test={len(vt)}"
        )
    else:
        logger.info("  ✓ Zero data leakage verified between splits")


# ---------------------------------------------------------------------------
# Task-specific preprocessors
# ---------------------------------------------------------------------------

def preprocess_translation(
    data_dir: Path,
    out_dir: Path,
    normalizer: CodeMixedNormalizer,
) -> None:
    """Preprocess Translation dataset (MT_train.csv / MT_test.csv)."""
    comi_dir = data_dir / "COMI-LINGUA"
    train_csv = comi_dir / "MT_train.csv"
    test_csv = comi_dir / "MT_test.csv"

    out_mt_dir = out_dir / "translation"
    out_mt_dir.mkdir(parents=True, exist_ok=True)

    if train_csv.exists() and test_csv.exists():
        logger.info("Preprocessing local COMI-LINGUA MT dataset...")
        
        def _parse_mt_df(path: Path) -> List[Dict[str, str]]:
            df = pd.read_csv(path)
            records = []
            for _, row in df.iterrows():
                src = str(row.get("Sentences", "")).strip()
                if not src or src == "nan":
                    continue

                # Target translation preference order: DH (Devanagari) -> RH (Romanized) -> Pred
                tgt = None
                for col in [
                    "Annotator_1_DH_translation",
                    "Annotator_2_DH_translation",
                    "Annotator_3_DH_translation",
                    "Predicted_DH_translation",
                    "Annotator_1_RH_translation",
                ]:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip() and str(val).strip() != "nan":
                        tgt = str(val).strip()
                        break

                if tgt:
                    records.append({
                        "source": normalizer.normalize(src),
                        "target": tgt,
                    })
            return records

        train_all = _parse_mt_df(train_csv)
        test_data = _parse_mt_df(test_csv)

        train_all = _deduplicate(train_all, "source")
        test_data = _deduplicate(test_data, "source")

        train_data, val_data = _split_data(train_all, train_ratio=0.9, val_ratio=0.1)

        _check_leakage(train_data, val_data, test_data, "source")
        _save_jsonl(train_data, out_mt_dir / "train.jsonl")
        _save_jsonl(val_data, out_mt_dir / "val.jsonl")
        _save_jsonl(test_data, out_mt_dir / "test.jsonl")
        logger.info(f"Translation preprocessing complete! Total: {len(train_data)+len(val_data)+len(test_data)}")

    else:
        logger.warning(f"Local MT CSV files not found in {comi_dir}. Run download_datasets.py or place CSV files there.")


def preprocess_ner(
    data_dir: Path,
    out_dir: Path,
    normalizer: CodeMixedNormalizer,
) -> None:
    """Preprocess NER dataset (NER_train.csv / NER_test.csv)."""
    comi_dir = data_dir / "COMI-LINGUA"
    train_csv = comi_dir / "NER_train.csv"
    test_csv = comi_dir / "NER_test.csv"

    out_ner_dir = out_dir / "ner"
    out_ner_dir.mkdir(parents=True, exist_ok=True)

    if train_csv.exists() and test_csv.exists():
        logger.info("Preprocessing local COMI-LINGUA NER dataset...")

        entity_map = {
            "ORGANISATION": "ORG",
            "PERSON": "PER",
            "LOCATION": "LOC",
            "MISCELLANEOUS": "MISC",
            "ORGANIZATION": "ORG",
        }

        def _parse_ner_df(path: Path) -> List[Dict[str, Any]]:
            df = pd.read_csv(path)
            records = []
            for _, row in df.iterrows():
                ann = row.get("Annotated by: Annotator 1")
                if pd.isna(ann) or not str(ann).strip():
                    ann = row.get("Annotated by: Annotator 2")
                if pd.isna(ann) or not str(ann).strip():
                    ann = row.get("Predicted tags")

                if pd.isna(ann) or not str(ann).strip():
                    continue

                try:
                    items = ast.literal_eval(str(ann))
                    tokens = []
                    bio_tags = []
                    prev_ent = None

                    for item in items:
                        w = item.get("word", "").strip()
                        ent = item.get("entity", "").strip().upper()
                        if not w:
                            continue
                        tokens.append(w)

                        if ent in ("X", "O", "", "NONE", "NULL"):
                            bio_tags.append("O")
                            prev_ent = None
                        else:
                            mapped = entity_map.get(ent, "MISC")
                            if mapped == prev_ent:
                                bio_tags.append(f"I-{mapped}")
                            else:
                                bio_tags.append(f"B-{mapped}")
                                prev_ent = mapped

                    if tokens and bio_tags:
                        records.append({
                            "tokens": tokens,
                            "ner_tags": bio_tags,
                        })
                except Exception:
                    continue
            return records

        train_all = _parse_ner_df(train_csv)
        test_data = _parse_ner_df(test_csv)

        train_data, val_data = _split_data(train_all, train_ratio=0.9, val_ratio=0.1)

        _save_jsonl(train_data, out_ner_dir / "train.jsonl")
        _save_jsonl(val_data, out_ner_dir / "val.jsonl")
        _save_jsonl(test_data, out_ner_dir / "test.jsonl")
        logger.info(f"NER preprocessing complete! Total: {len(train_data)+len(val_data)+len(test_data)}")

    else:
        logger.warning(f"Local NER CSV files not found in {comi_dir}.")


def preprocess_sentiment(
    data_dir: Path,
    out_dir: Path,
    normalizer: CodeMixedNormalizer,
) -> None:
    """Preprocess sentiment dataset or create synthetic baseline dataset if missing."""
    out_sent_dir = out_dir / "sentiment"
    out_sent_dir.mkdir(parents=True, exist_ok=True)

    sent_raw = data_dir / "raw" / "sentiment"
    all_data = []

    if sent_raw.exists():
        from datasets import load_from_disk
        for sub in sent_raw.iterdir():
            if sub.is_dir():
                try:
                    ds = load_from_disk(str(sub))
                    for split in ds:
                        for example in ds[split]:
                            text = example.get("text", example.get("sentence", ""))
                            label = example.get("label", example.get("sentiment", None))
                            if text and label is not None:
                                all_data.append({
                                    "text": normalizer.normalize(text),
                                    "label": int(label) if isinstance(label, (int, float)) else label,
                                })
                except Exception as e:
                    logger.warning(f"  Could not load {sub}: {e}")

    # Fallback to high-quality code-mixed sentiment seed examples if no dataset downloaded yet
    if not all_data:
        logger.info("  Generating seed Hinglish sentiment dataset...")
        seed_samples = [
            ("Yeh movie bohot acchi thi, cast and direction mind blowing tha!", 2),
            ("Khaana ekdum bekar tha, total waste of money and time.", 0),
            ("Flight timing normal tha, okayish service.", 1),
            ("Great experience! Subah se lekar raat tak enjoy kiya.", 2),
            ("Boht ganda performance, completely disappointed.", 0),
            ("Product delivery on time, package thik thak tha.", 1),
            ("Acha response mila support team se, thank you so much!", 2),
            ("Worst customer care ever, phone bhi attend nahi karte.", 0),
            ("Simple movie hai, light entertainment.", 1),
            ("Superb acting by the main lead, highly recommended!", 2),
            ("Kharab quality hai, 2 din me toot gaya.", 0),
            ("Avg speed hai internet ki, overall fine.", 1),
        ] * 50
        all_data = [{"text": normalizer.normalize(t), "label": l} for t, l in seed_samples]

    all_data = _deduplicate(all_data, "text")
    train_val, test_data = _split_data(all_data, train_ratio=0.85, val_ratio=0.15)
    train_data, val_data = _split_data(train_val, train_ratio=0.85, val_ratio=0.15)

    _check_leakage(train_data, val_data, test_data, "text")
    _save_jsonl(train_data, out_sent_dir / "train.jsonl")
    _save_jsonl(val_data, out_sent_dir / "val.jsonl")
    _save_jsonl(test_data, out_sent_dir / "test.jsonl")
    logger.info(f"Sentiment preprocessing complete! Total: {len(all_data)}")


def preprocess_qa(
    data_dir: Path,
    out_dir: Path,
    normalizer: CodeMixedNormalizer,
) -> None:
    """Preprocess QA dataset."""
    out_qa_dir = out_dir / "qa"
    out_qa_dir.mkdir(parents=True, exist_ok=True)

    qa_raw = data_dir / "raw" / "qa"
    all_data = []

    if qa_raw.exists():
        from datasets import load_from_disk
        for sub in qa_raw.iterdir():
            if sub.is_dir():
                try:
                    ds = load_from_disk(str(sub))
                    for split in ds:
                        for example in ds[split]:
                            context = example.get("context", "")
                            question = example.get("question", "")
                            answers = example.get("answers", {})
                            if context and question:
                                all_data.append({
                                    "context": normalizer.normalize(context),
                                    "question": normalizer.normalize(question),
                                    "answers": answers,
                                })
                except Exception as e:
                    logger.warning(f"  Could not load {sub}: {e}")

    if not all_data:
        logger.info("  Generating seed Hinglish QA dataset...")
        seed_qa = [
            (
                "Shah Rukh Khan Mumbai ke Mannat house me rehte hain. Woh Indian cinema ke megastar hain.",
                "Shah Rukh Khan kahan rehte hain?",
                {"text": ["Mumbai ke Mannat house"], "answer_start": [16]},
            ),
            (
                "Google ne Bangalore me naya R&D office open kiya hai.",
                "Google ne naya office kahan khola?",
                {"text": ["Bangalore"], "answer_start": [10]},
            ),
        ] * 50
        all_data = [
            {
                "context": normalizer.normalize(c),
                "question": normalizer.normalize(q),
                "answers": a,
            }
            for c, q, a in seed_qa
        ]

    all_data = _deduplicate(all_data, "question")
    train_val, test_data = _split_data(all_data, train_ratio=0.85, val_ratio=0.15)
    train_data, val_data = _split_data(train_val, train_ratio=0.85, val_ratio=0.15)

    _save_jsonl(train_data, out_qa_dir / "train.jsonl")
    _save_jsonl(val_data, out_qa_dir / "val.jsonl")
    _save_jsonl(test_data, out_qa_dir / "test.jsonl")
    logger.info(f"QA preprocessing complete! Total: {len(all_data)}")


def preprocess_intent(
    data_dir: Path,
    out_dir: Path,
    normalizer: CodeMixedNormalizer,
) -> None:
    """Preprocess intent dataset with synthetic code-mixing."""
    out_intent_dir = out_dir / "intent"
    out_intent_dir.mkdir(parents=True, exist_ok=True)

    mixer = SyntheticCodeMixer(replacement_prob=0.35, seed=42)

    atis_path = data_dir / "raw" / "intent" / "atis"
    texts = []
    labels = []
    
    if atis_path.exists():
        from datasets import load_from_disk
        try:
            ds = load_from_disk(str(atis_path))
            for split in ds:
                # To prevent excessive processing time in the demo, limit to 2000 per split
                # But ATIS train is ~4000 examples, so just use it all if possible
                for example in ds[split]:
                    text = example.get('text', '')
                    intent = example.get('intent', '')
                    if text and intent is not None:
                        texts.append(text)
                        labels.append(intent)
        except Exception as e:
            logger.warning(f"Could not load ATIS from disk: {e}")

    if not texts:
        logger.warning("Falling back to small synthetic dummy intents")
        base_intents = [
            ("I want to book a flight ticket from Delhi to Mumbai tomorrow.", "book_flight"),
            ("What is the weather forecast for Bangalore today?", "get_weather"),
            ("Show me my current account balance.", "check_balance"),
            ("Can you order food from the nearest restaurant?", "order_food"),
            ("Please cancel my train booking right now.", "cancel_booking"),
        ] * 60
        texts = [t for t, l in base_intents]
        labels = [l for t, l in base_intents]

    logger.info(f"Applying synthetic code-mixing to {len(texts)} queries...")
    mixed_texts, mixed_labels = mixer.code_mix_batch(texts, labels, num_variants=2)
    all_data = [
        {"text": normalizer.normalize(t), "intent": l}
        for t, l in zip(mixed_texts, mixed_labels)
    ]

    all_data = _deduplicate(all_data, "text")
    train_val, test_data = _split_data(all_data, train_ratio=0.85, val_ratio=0.15)
    train_data, val_data = _split_data(train_val, train_ratio=0.85, val_ratio=0.15)

    _save_jsonl(train_data, out_intent_dir / "train.jsonl")
    _save_jsonl(val_data, out_intent_dir / "val.jsonl")
    _save_jsonl(test_data, out_intent_dir / "test.jsonl")
    logger.info(f"Intent preprocessing complete! Total: {len(all_data)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "translation", "ner", "sentiment", "qa", "intent"],
    )
    parser.add_argument(
        "--lang-pair",
        default="hi-en",
        help="Language pair for normalization (default: hi-en)",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_dir = resolve_path(cfg, "data_dir")
    out_dir = resolve_path(cfg, "processed_data_dir")

    normalizer = CodeMixedNormalizer(lang_pair=args.lang_pair)

    tasks = args.tasks
    if "all" in tasks:
        tasks = ["translation", "ner", "sentiment", "qa", "intent"]

    logger.info(f"Starting preprocessing for tasks: {tasks}")
    logger.info(f"Data root: {data_dir}")

    if "translation" in tasks:
        preprocess_translation(data_dir, out_dir, normalizer)
    if "ner" in tasks:
        preprocess_ner(data_dir, out_dir, normalizer)
    if "sentiment" in tasks:
        preprocess_sentiment(data_dir, out_dir, normalizer)
    if "qa" in tasks:
        preprocess_qa(data_dir, out_dir, normalizer)
    if "intent" in tasks:
        preprocess_intent(data_dir, out_dir, normalizer)

    logger.info("All task datasets preprocessed successfully!")


if __name__ == "__main__":
    main()
