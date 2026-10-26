"""
Dataset downloader — fetches all required datasets from Hugging Face and
other sources, saving raw data to ``data/raw/``.

Datasets downloaded:
1. COMI-LINGUA (NER, Translation, POS, LID, MLI)
2. Sentiment Analysis datasets (L3Cube-HingSent or fallbacks)
3. Question Answering (IndicQA Hindi split or SQuAD)
4. Intent Detection (ATIS — to be code-mixed synthetically)

Usage:
    python data/scripts/download_datasets.py
    python data/scripts/download_datasets.py --tasks sentiment ner
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_comi_lingua(raw_dir: Path) -> None:
    """Download COMI-LINGUA dataset from Hugging Face."""
    from datasets import load_dataset

    logger.info("Downloading COMI-LINGUA from Hugging Face...")
    save_path = raw_dir / "comi_lingua"
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset("LingoIITGN/COMI-LINGUA")
        ds.save_to_disk(str(save_path))
        logger.info(f"COMI-LINGUA saved to {save_path}")

        # Log dataset info
        for split_name, split_data in ds.items():
            logger.info(f"  {split_name}: {len(split_data)} examples")
            logger.info(f"  Columns: {split_data.column_names}")
    except Exception as e:
        logger.error(f"Failed to download COMI-LINGUA: {e}")
        logger.info("You can manually download from: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA")


def download_sentiment_data(raw_dir: Path, cfg: dict) -> None:
    """Download sentiment analysis datasets."""
    from datasets import load_dataset

    save_path = raw_dir / "sentiment"
    save_path.mkdir(parents=True, exist_ok=True)

    repos = ["airzipm/sentiment-dataset-en-hi-hinglish-v2"]
    repos.extend(cfg["datasets"]["sentiment"].get("fallback_repos", []))

    for repo in repos:
        logger.info(f"Trying sentiment dataset: {repo}")
        try:
            ds = load_dataset(repo)
            ds.save_to_disk(str(save_path / repo.replace("/", "_")))
            logger.info(f"Sentiment dataset saved: {repo}")
            for split_name, split_data in ds.items():
                logger.info(f"  {split_name}: {len(split_data)} examples")
            return  # Success — stop trying fallbacks
        except Exception as e:
            logger.warning(f"Could not download {repo}: {e}")

    logger.error("No sentiment dataset could be downloaded. Manual setup required.")


def download_qa_data(raw_dir: Path, cfg: dict) -> None:
    """Download question answering datasets."""
    from datasets import load_dataset

    save_path = raw_dir / "qa"
    save_path.mkdir(parents=True, exist_ok=True)

    repos = [cfg["datasets"]["qa"]["hf_repo"]]
    repos.extend(cfg["datasets"]["qa"].get("fallback_repos", []))

    for repo in repos:
        logger.info(f"Trying QA dataset: {repo}")
        try:
            if repo == "ai4bharat/IndicQA":
                ds = load_dataset(repo, "indicqa.hi")
            elif repo == "squad_v2":
                ds = load_dataset(repo)
            else:
                ds = load_dataset(repo)
            ds.save_to_disk(str(save_path / repo.replace("/", "_")))
            logger.info(f"QA dataset saved: {repo}")
            for split_name, split_data in ds.items():
                logger.info(f"  {split_name}: {len(split_data)} examples")
            return
        except Exception as e:
            logger.warning(f"Could not download {repo}: {e}")

    logger.error("No QA dataset could be downloaded. Manual setup required.")


def download_intent_data(raw_dir: Path) -> None:
    """Download ATIS intent detection dataset (English base).

    This will later be code-mixed synthetically during preprocessing.
    """
    from datasets import load_dataset

    save_path = raw_dir / "intent"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ATIS dataset for intent detection...")
    try:
        # Try the standard ATIS dataset
        ds = load_dataset("tuetschek/atis")
        ds.save_to_disk(str(save_path / "atis"))
        logger.info(f"ATIS dataset saved to {save_path / 'atis'}")
        for split_name, split_data in ds.items():
            logger.info(f"  {split_name}: {len(split_data)} examples")
    except Exception as e:
        logger.warning(f"Could not download ATIS: {e}")
        logger.info("Trying SNIPS as fallback...")
        try:
            ds = load_dataset("benayas/snips")
            ds.save_to_disk(str(save_path / "snips"))
            logger.info(f"SNIPS dataset saved to {save_path / 'snips'}")
        except Exception as e2:
            logger.error(f"Could not download SNIPS either: {e2}")
            logger.info("Intent dataset must be provided manually.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets for code-mixed NLP")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        choices=["all", "comi_lingua", "sentiment", "qa", "intent"],
        help="Which datasets to download",
    )
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = resolve_path(cfg, "raw_data_dir")

    tasks = args.tasks
    if "all" in tasks:
        tasks = ["comi_lingua", "sentiment", "qa", "intent"]

    logger.info(f"Downloading datasets: {tasks}")
    logger.info(f"Raw data directory: {raw_dir}")

    if "comi_lingua" in tasks:
        download_comi_lingua(raw_dir)
    if "sentiment" in tasks:
        download_sentiment_data(raw_dir, cfg)
    if "qa" in tasks:
        download_qa_data(raw_dir, cfg)
    if "intent" in tasks:
        download_intent_data(raw_dir)

    logger.info("Dataset download complete!")


if __name__ == "__main__":
    main()
