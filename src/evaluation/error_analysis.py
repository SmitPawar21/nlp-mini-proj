"""
Qualitative error analysis utilities.

Categorises prediction errors by type for code-mixed NLP tasks:
- Language-switch boundary errors (model fails at code-switching points)
- Script confusion (Devanagari vs Romanized)
- Entity boundary errors (NER partial matches)
- Fluency errors (translation)
- Out-of-vocabulary errors

Usage:
    from src.evaluation.error_analysis import analyse_errors
    report = analyse_errors("sentiment", predictions, references, texts)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.lang_utils import detect_script, ScriptInfo

logger = logging.getLogger(__name__)


@dataclass
class ErrorSample:
    """A single error instance with context."""
    text: str
    prediction: Any
    reference: Any
    error_type: str
    script_info: Optional[ScriptInfo] = None
    notes: str = ""


@dataclass
class ErrorReport:
    """Summary of error analysis."""
    task: str
    total_errors: int = 0
    total_samples: int = 0
    error_rate: float = 0.0
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    samples_by_type: Dict[str, List[ErrorSample]] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"=== Error Analysis Report: {self.task} ===",
            f"Total samples: {self.total_samples}",
            f"Total errors: {self.total_errors} ({self.error_rate:.1%})",
            "",
            "Error type breakdown:",
        ]
        for etype, count in sorted(
            self.error_type_counts.items(), key=lambda x: -x[1]
        ):
            pct = count / self.total_errors * 100 if self.total_errors else 0
            lines.append(f"  {etype}: {count} ({pct:.1f}%)")

        lines.append("")
        lines.append("Sample errors per category:")
        for etype, samples in self.samples_by_type.items():
            lines.append(f"\n--- {etype} ---")
            for s in samples[:3]:  # Show top 3 per category
                lines.append(f"  Text: {s.text[:100]}...")
                lines.append(f"  Pred: {s.prediction}")
                lines.append(f"  True: {s.reference}")
                if s.notes:
                    lines.append(f"  Note: {s.notes}")
                lines.append("")

        return "\n".join(lines)


def _classify_classification_error(
    text: str, pred: Any, ref: Any
) -> str:
    """Classify a classification error by heuristic type."""
    info = detect_script(text)

    if info.is_mixed_script:
        return "language_switch_boundary"
    elif info.dominant_script == "devanagari":
        return "devanagari_input"
    elif len(text.split()) <= 3:
        return "short_text"
    else:
        return "general_misclassification"


def _classify_ner_error(
    tokens: List[str], pred_tags: List[str], ref_tags: List[str]
) -> str:
    """Classify NER error by type."""
    # Check for partial entity match
    pred_entities = _extract_entity_spans(pred_tags)
    ref_entities = _extract_entity_spans(ref_tags)

    if not pred_entities and ref_entities:
        return "missed_entity"
    elif pred_entities and not ref_entities:
        return "hallucinated_entity"

    # Check for boundary errors (partial overlap)
    for ref_start, ref_end, ref_type in ref_entities:
        for pred_start, pred_end, pred_type in pred_entities:
            if (
                (pred_start <= ref_end and pred_end >= ref_start)
                and (pred_start != ref_start or pred_end != ref_end)
            ):
                return "entity_boundary_error"
            if (
                pred_start == ref_start
                and pred_end == ref_end
                and pred_type != ref_type
            ):
                return "entity_type_error"

    return "general_ner_error"


def _extract_entity_spans(
    tags: List[str],
) -> List[tuple]:
    """Extract (start, end, type) spans from BIO tags."""
    spans = []
    current_start = None
    current_type = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_start is not None:
                spans.append((current_start, i - 1, current_type))
            current_start = i
            current_type = tag[2:]
        elif tag.startswith("I-"):
            if current_start is None:
                current_start = i
                current_type = tag[2:]
        else:
            if current_start is not None:
                spans.append((current_start, i - 1, current_type))
                current_start = None
                current_type = None

    if current_start is not None:
        spans.append((current_start, len(tags) - 1, current_type))

    return spans


def analyse_errors(
    task: str,
    predictions: List[Any],
    references: List[Any],
    texts: List[str],
    max_samples_per_type: int = 10,
) -> ErrorReport:
    """Perform error analysis for a given task.

    Parameters
    ----------
    task : str
        Task name.
    predictions : list
        Model predictions.
    references : list
        Ground truth labels.
    texts : list of str
        Input texts (for context).
    max_samples_per_type : int
        Max error samples to store per error category.

    Returns
    -------
    ErrorReport
    """
    report = ErrorReport(task=task, total_samples=len(predictions))
    error_counts: Dict[str, int] = {}
    error_samples: Dict[str, List[ErrorSample]] = {}

    for text, pred, ref in zip(texts, predictions, references):
        if pred == ref:
            continue

        # Classify the error
        if task in ("sentiment", "intent"):
            error_type = _classify_classification_error(text, pred, ref)
        elif task == "ner":
            error_type = _classify_ner_error(text.split(), pred, ref)
        elif task == "translation":
            info = detect_script(text)
            error_type = "mixed_script_translation" if info.is_mixed_script else "general_translation"
        elif task == "qa":
            error_type = "answer_extraction"
        else:
            error_type = "unknown"

        error_counts[error_type] = error_counts.get(error_type, 0) + 1

        if error_type not in error_samples:
            error_samples[error_type] = []
        if len(error_samples[error_type]) < max_samples_per_type:
            error_samples[error_type].append(
                ErrorSample(
                    text=text,
                    prediction=pred,
                    reference=ref,
                    error_type=error_type,
                    script_info=detect_script(text),
                )
            )

    report.total_errors = sum(error_counts.values())
    report.error_rate = report.total_errors / report.total_samples if report.total_samples else 0
    report.error_type_counts = error_counts
    report.samples_by_type = error_samples

    return report
