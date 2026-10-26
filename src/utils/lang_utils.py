"""
Language and script detection utilities for code-mixed text.

Provides lightweight, rule-based detectors for:
- Script identification (Devanagari, Latin, mixed)
- Language-mix ratio estimation
- Token-level language tagging heuristics

Designed to be extensible: add new Unicode ranges for additional scripts
(e.g., Tamil, Bengali) without changing the public API.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Unicode block ranges for script detection
# ---------------------------------------------------------------------------
_SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],  # Devanagari + extended
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A)],        # Basic Latin A-Z, a-z
    "arabic": [(0x0600, 0x06FF)],                          # For future Urdu support
    "tamil": [(0x0B80, 0x0BFF)],                           # For future Tamil support
    "bengali": [(0x0980, 0x09FF)],                         # For future Bengali support
    "digit": [(0x0030, 0x0039)],                           # 0-9
}


def _char_script(ch: str) -> str:
    """Return the script name for a single character."""
    cp = ord(ch)
    for script, ranges in _SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    return "other"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ScriptInfo:
    """Result of script analysis on a text string."""

    dominant_script: str
    script_counts: Dict[str, int] = field(default_factory=dict)
    is_mixed_script: bool = False
    mix_ratio: float = 0.0  # ratio of non-dominant script characters

    def __repr__(self) -> str:
        return (
            f"ScriptInfo(dominant={self.dominant_script!r}, "
            f"mixed={self.is_mixed_script}, ratio={self.mix_ratio:.2f})"
        )


@dataclass
class LanguageMixInfo:
    """Result of language-mix analysis at the token level."""

    hi_ratio: float = 0.0
    en_ratio: float = 0.0
    other_ratio: float = 0.0
    num_tokens: int = 0
    is_code_mixed: bool = False


def detect_script(text: str) -> ScriptInfo:
    """Analyse the script composition of *text*.

    Parameters
    ----------
    text : str
        Input text (can be code-mixed).

    Returns
    -------
    ScriptInfo
        Dominant script, character-level counts, and mixing information.
    """
    counts: Counter = Counter()
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        counts[_char_script(ch)] += 1

    # Remove 'other' and 'digit' from dominant-script calculation
    script_counts = {k: v for k, v in counts.items() if k not in ("other", "digit")}
    total = sum(script_counts.values()) or 1

    if not script_counts:
        return ScriptInfo(dominant_script="unknown", script_counts=dict(counts))

    dominant = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
    non_dominant = total - script_counts[dominant]
    mix_ratio = non_dominant / total

    return ScriptInfo(
        dominant_script=dominant,
        script_counts=dict(counts),
        is_mixed_script=mix_ratio > 0.1,  # >10% from another script
        mix_ratio=round(mix_ratio, 4),
    )


def detect_language_mix(
    tokens: List[str],
    hi_vocab: Optional[set] = None,
    en_vocab: Optional[set] = None,
) -> LanguageMixInfo:
    """Estimate Hindi vs English token ratio using script heuristics.

    A token is classified as Hindi if it contains ≥1 Devanagari character,
    English if all alphabetic characters are Latin, otherwise other.
    Optional vocabulary sets can refine classification for Romanized Hindi.

    Parameters
    ----------
    tokens : list of str
        Pre-tokenised words.
    hi_vocab : set, optional
        Known Romanized Hindi words for better classification.
    en_vocab : set, optional
        Known English words for disambiguation.

    Returns
    -------
    LanguageMixInfo
    """
    if not tokens:
        return LanguageMixInfo()

    hi_count = en_count = other_count = 0

    for tok in tokens:
        tok_lower = tok.lower().strip()
        if not tok_lower:
            continue

        # Check Devanagari presence first
        has_dev = any(_char_script(ch) == "devanagari" for ch in tok_lower)
        has_lat = any(_char_script(ch) == "latin" for ch in tok_lower)

        if has_dev:
            hi_count += 1
        elif has_lat:
            # If we have vocab lists, use them for Romanized Hindi
            if hi_vocab and tok_lower in hi_vocab:
                hi_count += 1
            elif en_vocab and tok_lower in en_vocab:
                en_count += 1
            else:
                en_count += 1  # Default: Latin script → English
        else:
            other_count += 1

    total = hi_count + en_count + other_count or 1
    hi_r = hi_count / total
    en_r = en_count / total
    ot_r = other_count / total

    return LanguageMixInfo(
        hi_ratio=round(hi_r, 4),
        en_ratio=round(en_r, 4),
        other_ratio=round(ot_r, 4),
        num_tokens=len(tokens),
        is_code_mixed=(min(hi_r, en_r) > 0.1),  # Both languages ≥10%
    )


def is_devanagari(text: str) -> bool:
    """Return True if the majority of alphabetic chars are Devanagari."""
    info = detect_script(text)
    return info.dominant_script == "devanagari"


def is_romanized(text: str) -> bool:
    """Return True if text is predominantly Latin script (possibly Romanized Hindi)."""
    info = detect_script(text)
    return info.dominant_script == "latin"
