"""
CLI Predict Script — Run inference on single text inputs or files.

Usage:
    python scripts/predict.py --task sentiment --text "Yeh movie bohot achhi thi"
    python scripts/predict.py --task translation --text "Fast solution nikalna padega"
    python scripts/predict.py --task qa --text "Shah Rukh Khan kahan rehte hain?" --context "Shah Rukh Khan Mumbai me rehte hain."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import UnifiedPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on code-mixed text")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["translation", "sentiment", "ner", "qa", "intent"],
        help="NLP task",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input code-mixed text or question",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Context paragraph (required for QA)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing trained model weights",
    )

    args = parser.parse_args()

    predictor = UnifiedPredictor(models_dir=args.model_dir)
    result = predictor.predict(
        text=args.text,
        task=args.task,
        context=args.context,
    )

    print("\n" + "=" * 50)
    print(f"Task             : {result['task'].upper()}")
    print(f"Original Input   : {result['raw_input']}")
    print(f"Normalized Input : {result['normalized_input']}")
    print(f"Dominant Script  : {result['dominant_script']}")
    print(f"Mixed Script?    : {result['is_mixed_script']}")
    print("-" * 50)

    if args.task == "sentiment":
        print(f"Sentiment  : {result['sentiment']}")
        print(f"Confidence : {result['confidence']:.2%}")
    elif args.task == "translation":
        print(f"Translation: {result['translation']}")
    elif args.task == "ner":
        print("Entities   :")
        for ent in result.get("entities", []):
            print(f"  - [{ent['entity']}] {ent['word']} (score: {ent['score']:.2%})")
    elif args.task == "qa":
        print(f"Answer     : {result['answer']}")
        print(f"Confidence : {result['confidence']:.2%}")
    elif args.task == "intent":
        print(f"Intent     : {result['intent']}")
        print(f"Confidence : {result['confidence']:.2%}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
