"""
Code-mixed text normalizer.

Handles the specific challenges of Hindi-English code-mixed data:
- Unicode normalization (NFC)
- Romanized Hindi spelling variations
- Social-media noise (URLs, @mentions, hashtags, emojis)
- Character elongation reduction ("soooo gooood" → "soo good")
- Punctuation normalization

Parameterised by language pair so new languages can be added by extending
the romanization maps and regex patterns without touching core logic.

Usage:
    from src.preprocessing.normalizer import CodeMixedNormalizer
    normalizer = CodeMixedNormalizer(lang_pair="hi-en")
    clean = normalizer.normalize("Yeh bahut acchaaaaa hai @user https://t.co/x")
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Common Romanized Hindi spelling variations → canonical form
# Extend this dict when adding new languages.
# ---------------------------------------------------------------------------
_HINDI_ROMAN_MAP: Dict[str, str] = {
    # Common variations of Hindi words in Roman script
    "accha": "acha",
    "acchaa": "acha",
    "achha": "acha",
    "achcha": "acha",
    "bahut": "bahut",
    "bohot": "bahut",
    "bohut": "bahut",
    "kya": "kya",
    "kyaa": "kya",
    "hai": "hai",
    "hain": "hain",
    "nahi": "nahi",
    "nahin": "nahi",
    "nhi": "nahi",
    "mein": "mein",
    "mai": "mein",
    "hum": "hum",
    "ham": "hum",
    "aur": "aur",
    "or": "aur",
    "lekin": "lekin",
    "lekn": "lekin",
    "par": "par",
    "per": "par",
    "toh": "to",
    "tho": "to",
    "yeh": "yeh",
    "ye": "yeh",
    "woh": "woh",
    "wo": "woh",
    "tha": "tha",
    "thaa": "tha",
    "raha": "raha",
    "rha": "raha",
    "kar": "kar",
    "krr": "kar",
    "kr": "kar",
    "zyada": "zyada",
    "jyada": "zyada",
    "jayda": "zyada",
    "sab": "sab",
    "sub": "sab",
    "kuch": "kuch",
    "kuchh": "kuch",
    "koi": "koi",
    "koii": "koi",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|t\.co/\S+", re.IGNORECASE
)
_USERNAME_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")
_MULTI_SPACE = re.compile(r"\s+")
_MULTI_PUNCT = re.compile(r"([!?.]){2,}")


class CodeMixedNormalizer:
    """Normalize code-mixed text for NLP tasks.

    Parameters
    ----------
    lang_pair : str
        Language pair identifier (e.g. "hi-en"). Controls which
        romanization map is used.
    remove_urls : bool
        Replace URLs with empty string.
    remove_usernames : bool
        Replace @mentions with empty string.
    remove_emojis : bool
        Strip emoji characters.
    keep_hashtag_text : bool
        If True, keep hashtag text but remove the ``#`` symbol.
    max_elongation : int
        Maximum consecutive repeated characters allowed.
    normalize_roman : bool
        Apply Romanized Hindi spelling normalization.
    """

    def __init__(
        self,
        lang_pair: str = "hi-en",
        remove_urls: bool = True,
        remove_usernames: bool = True,
        remove_emojis: bool = False,
        keep_hashtag_text: bool = True,
        max_elongation: int = 3,
        normalize_roman: bool = True,
    ) -> None:
        self.lang_pair = lang_pair
        self.remove_urls = remove_urls
        self.remove_usernames = remove_usernames
        self.remove_emojis = remove_emojis
        self.keep_hashtag_text = keep_hashtag_text
        self.max_elongation = max_elongation
        self.normalize_roman = normalize_roman

        # Select romanization map based on language pair
        self._roman_map = self._get_roman_map(lang_pair)

    @staticmethod
    def _get_roman_map(lang_pair: str) -> Dict[str, str]:
        """Return the romanization normalization map for a language pair.

        Extend this method when adding new language pairs.
        """
        maps = {
            "hi-en": _HINDI_ROMAN_MAP,
            # Future: "ta-en": _TAMIL_ROMAN_MAP,
            # Future: "bn-en": _BENGALI_ROMAN_MAP,
        }
        return maps.get(lang_pair, {})

    # ------------------------------------------------------------------
    # Individual normalization steps
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Apply NFC Unicode normalization."""
        return unicodedata.normalize("NFC", text)

    def clean_urls(self, text: str) -> str:
        """Remove or replace URLs."""
        if self.remove_urls:
            return _URL_PATTERN.sub("", text)
        return _URL_PATTERN.sub("<URL>", text)

    def clean_usernames(self, text: str) -> str:
        """Remove or replace @mentions."""
        if self.remove_usernames:
            return _USERNAME_PATTERN.sub("", text)
        return _USERNAME_PATTERN.sub("<USER>", text)

    def clean_hashtags(self, text: str) -> str:
        """Process hashtags — keep text or remove entirely."""
        if self.keep_hashtag_text:
            return _HASHTAG_PATTERN.sub(r"\1", text)
        return _HASHTAG_PATTERN.sub("", text)

    def clean_emojis(self, text: str) -> str:
        """Remove emoji characters if configured."""
        if self.remove_emojis:
            return _EMOJI_PATTERN.sub("", text)
        return text

    def reduce_elongation(self, text: str) -> str:
        """Reduce character repetitions beyond *max_elongation*.

        ``"goooooood"`` → ``"goood"`` (with max_elongation=3).
        """
        def _replace(match: re.Match) -> str:
            return match.group(1) * min(len(match.group(0)), self.max_elongation)

        return _ELONGATION_PATTERN.sub(_replace, text)

    def normalize_romanized(self, text: str) -> str:
        """Map known Romanized Hindi spelling variants to canonical forms."""
        if not self.normalize_roman or not self._roman_map:
            return text
        tokens = text.split()
        normalised = []
        for tok in tokens:
            key = tok.lower()
            if key in self._roman_map:
                # Preserve original casing pattern for first char
                replacement = self._roman_map[key]
                if tok[0].isupper():
                    replacement = replacement.capitalize()
                normalised.append(replacement)
            else:
                normalised.append(tok)
        return " ".join(normalised)

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Collapse repeated punctuation (``!!!`` → ``!``)."""
        return _MULTI_PUNCT.sub(r"\1", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces and strip."""
        return _MULTI_SPACE.sub(" ", text).strip()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> str:
        """Run the full normalization pipeline.

        Order matters: URLs/usernames first (contain special chars),
        then character-level cleaning, then token-level normalization.
        """
        text = self.normalize_unicode(text)
        text = self.clean_urls(text)
        text = self.clean_usernames(text)
        text = self.clean_hashtags(text)
        text = self.clean_emojis(text)
        text = self.reduce_elongation(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_romanized(text)
        text = self.normalize_whitespace(text)
        return text

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a list of texts."""
        return [self.normalize(t) for t in texts]
