"""
CLI Evaluation Script — Evaluate trained or baseline models on test data.

Usage:
    python scripts/evaluate.py --task sentiment
    python scripts/evaluate.py --task ner --model-dir models/ner
    python scripts/evaluate.py --task translation --baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics
from src.evaluation.error_analysis import analyse_errors
from src.inference.predictor import UnifiedPredictor
from src.utils.config import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(processed_dir: Path, task: str) -> list[dict]:
    test_file = processed_dir / task / "test.jsonl"
    if not test_file.exists():
        logger.error(f"Test dataset file not found: {test_file}")
        sys.exit(1)
    
    data = []
    with open(test_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate code-mixed NLP models")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["translation", "sentiment", "ner", "qa", "intent"],
        help="NLP task to evaluate",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory of trained model",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save evaluation report to disk",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum test samples to evaluate (default: evaluate all)",
    )

    args = parser.parse_args()

    cfg = load_config(args.task)
    processed_dir = resolve_path(cfg, "processed_data_dir")
    test_data = load_test_data(processed_dir, args.task)
    if args.max_samples is not None:
        test_data = test_data[:args.max_samples]
    
    logger.info(f"Evaluating task '{args.task}' on {len(test_data)} test instances...")

    predictor = UnifiedPredictor(models_dir=args.model_dir)

    preds = []
    refs = []
    texts = []

    for item in test_data:
        if args.task == "sentiment":
            txt = item.get("text", "")
            ref_raw = item.get("label")
            # Map numeric label to string name
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            ref = label_map.get(ref_raw, str(ref_raw)) if isinstance(ref_raw, int) else str(ref_raw)
            res = predictor.predict(txt, task="sentiment")
            preds.append(res["sentiment"])
            refs.append(ref)
            texts.append(txt)

        elif args.task == "translation":
            txt = item.get("source", "")
            ref = item.get("target", "")
            res = predictor.predict(txt, task="translation")
            preds.append(res["translation"])
            refs.append(ref)
            texts.append(txt)

        elif args.task == "ner":
            tokens = item.get("tokens", [])
            txt = " ".join(tokens)
            ref_tags = item.get("ner_tags", [])
            res = predictor.predict(txt, task="ner")
            pred_tags = res.get("token_tags", [])
            # Pad or truncate pred_tags to match ref_tags length
            if len(pred_tags) < len(ref_tags):
                pred_tags.extend(["O"] * (len(ref_tags) - len(pred_tags)))
            else:
                pred_tags = pred_tags[:len(ref_tags)]
            preds.append(pred_tags)
            refs.append(ref_tags)
            texts.append(txt)

        elif args.task == "qa":
            ctx = item.get("context", "")
            q = item.get("question", "")
            ans = item.get("answers", {}).get("text", [""])[0] if isinstance(item.get("answers"), dict) else ""
            res = predictor.predict(q, task="qa", context=ctx)
            preds.append(res["answer"])
            refs.append(ans)
            texts.append(q)

        elif args.task == "intent":
            txt = item.get("text", "")
            ref = item.get("intent")
            res = predictor.predict(txt, task="intent")
            preds.append(res["intent"])
            refs.append(ref)
            texts.append(txt)

    # Compute metrics
    metrics = compute_metrics(args.task, preds, refs)
    logger.info(f"\n=== Evaluation Metrics [{args.task}] ===")
    for k, v in metrics.items():
        if k != "report" and k != "confusion_matrix":
            logger.info(f"  {k}: {v}")

    if "report" in metrics:
        logger.info(f"\n{metrics['report']}")

    # Error analysis
    error_rep = analyse_errors(args.task, preds, refs, texts)
    logger.info(f"\n{error_rep.summary()}")

    if args.save_report:
        output_dir = resolve_path(cfg, "output_dir")
        report_file = output_dir / f"eval_report_{args.task}.json"
        with open(report_file, "w", encoding="utf-8") as fh:
            json.dump({
                "task": args.task,
                "metrics": {k: v for k, v in metrics.items() if k != "report"},
                "error_summary": {
                    "total_errors": error_rep.total_errors,
                    "error_rate": error_rep.error_rate,
                    "type_counts": error_rep.error_type_counts,
                }
            }, fh, indent=2)
        logger.info(f"Evaluation report saved to {report_file}")


if __name__ == "__main__":
    main()
