"""
Data augmentation for code-mixed NLP tasks.

Provides:
- Back-translation augmentation (monolingual → code-mixed or vice versa)
- Synthetic code-mixing by injecting Hindi tokens into English text
- Random script switching (Devanagari ↔ Romanized) for robustness

These are especially important for low-resource tasks like Intent Detection
where no standard Hinglish benchmark exists.

Usage:
    from src.preprocessing.augmentation import BackTranslationAugmenter
    aug = BackTranslationAugmenter()
    augmented = aug.augment("What is the weather today?")
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple


class BackTranslationAugmenter:
    """Augment text via back-translation using a Hugging Face translation model.

    Translates English → Hindi → English (or code-mixed) to generate
    paraphrases that naturally introduce Hindi tokens.

    The translation model is loaded lazily on first use to avoid
    import-time GPU allocation.

    Parameters
    ----------
    model_name : str
        Hugging Face model for translation.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-hi",
        reverse_model_name: str = "Helsinki-NLP/opus-mt-hi-en",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.reverse_model_name = reverse_model_name
        self.device = device
        self._pipeline = None
        self._reverse_pipeline = None

    def _load_pipelines(self) -> None:
        """Lazy-load translation pipelines."""
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "translation",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
        )
        self._reverse_pipeline = pipeline(
            "translation",
            model=self.reverse_model_name,
            device=0 if self.device == "cuda" else -1,
        )

    def augment(self, text: str) -> str:
        """Generate an augmented version via back-translation.

        English → Hindi → English produces natural paraphrases.
        """
        self._load_pipelines()
        # Forward: English → Hindi
        hi = self._pipeline(text, max_length=256)[0]["translation_text"]
        # Reverse: Hindi → English
        back = self._reverse_pipeline(hi, max_length=256)[0]["translation_text"]
        return back

    def augment_batch(self, texts: List[str], num_augments: int = 1) -> List[str]:
        """Augment a batch of texts, returning originals + augmented copies."""
        augmented = list(texts)
        for _ in range(num_augments):
            for text in texts:
                augmented.append(self.augment(text))
        return augmented


class SyntheticCodeMixer:
    """Create synthetic code-mixed text by injecting Hindi words into English.

    Uses a simple dictionary-based replacement strategy. This is useful for
    creating training data for intent detection where no code-mixed dataset
    exists.

    Parameters
    ----------
    replacement_prob : float
        Probability of replacing an English word with its Hindi equivalent.
    """

    # Common English → Romanized Hindi word substitutions
    _SUBSTITUTION_MAP: Dict[str, List[str]] = {
        "what": ["kya"],
        "where": ["kahan", "kidhar"],
        "when": ["kab"],
        "how": ["kaise", "kaisa"],
        "why": ["kyun", "kyu"],
        "yes": ["haan", "ha"],
        "no": ["nahi", "nhi"],
        "good": ["acha", "badhiya"],
        "bad": ["bura", "kharab"],
        "is": ["hai"],
        "are": ["hain"],
        "want": ["chahiye", "chahte"],
        "need": ["chahiye", "zarurat"],
        "go": ["jao", "chalo"],
        "come": ["aao", "aana"],
        "tell": ["batao", "bolo"],
        "see": ["dekho", "dekhna"],
        "give": ["do", "dena"],
        "please": ["please", "kripya"],
        "thank": ["dhanyavaad", "shukriya"],
        "you": ["tum", "aap"],
        "i": ["mai", "mein"],
        "my": ["mera", "meri"],
        "this": ["yeh", "ye"],
        "that": ["woh", "wo"],
        "very": ["bahut", "bohot"],
        "also": ["bhi"],
        "but": ["lekin", "par"],
        "and": ["aur"],
        "with": ["ke saath"],
        "from": ["se"],
        "today": ["aaj"],
        "tomorrow": ["kal"],
        "now": ["abhi"],
        "here": ["yahan"],
        "there": ["wahan"],
        "food": ["khana"],
        "water": ["paani"],
        "money": ["paise", "paisa"],
        "time": ["samay", "waqt"],
        "work": ["kaam"],
        "home": ["ghar"],
        "friend": ["dost", "yaar"],
        "brother": ["bhai"],
        "sister": ["behen"],
    }

    def __init__(self, replacement_prob: float = 0.3, seed: int = 42) -> None:
        self.replacement_prob = replacement_prob
        self._rng = random.Random(seed)

    def code_mix(self, text: str) -> str:
        """Replace some English words with Hindi equivalents.

        Parameters
        ----------
        text : str
            English input text.

        Returns
        -------
        str
            Code-mixed version of the text.
        """
        tokens = text.split()
        mixed = []
        for tok in tokens:
            # Strip punctuation for lookup, preserve it
            clean = tok.lower().strip(".,!?;:\"'()")
            trailing = tok[len(clean):] if len(tok) > len(clean) else ""
            leading = tok[:len(tok) - len(clean) - len(trailing)] if len(tok) > len(clean) + len(trailing) else ""

            if (
                clean in self._SUBSTITUTION_MAP
                and self._rng.random() < self.replacement_prob
            ):
                hindi_options = self._SUBSTITUTION_MAP[clean]
                replacement = self._rng.choice(hindi_options)
                mixed.append(f"{leading}{replacement}{trailing}")
            else:
                mixed.append(tok)
        return " ".join(mixed)

    def code_mix_batch(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        num_variants: int = 2,
    ) -> Tuple[List[str], Optional[List[Any]]]:
        """Generate multiple code-mixed variants for each text.

        Parameters
        ----------
        texts : list of str
            Original English texts.
        labels : list, optional
            Corresponding labels (preserved for each variant).
        num_variants : int
            Number of code-mixed variants per input.

        Returns
        -------
        tuple of (list of str, list or None)
            Augmented texts and corresponding labels.
        """
        aug_texts = list(texts)
        aug_labels = list(labels) if labels else None

        for i, text in enumerate(texts):
            for _ in range(num_variants):
                mixed = self.code_mix(text)
                aug_texts.append(mixed)
                if aug_labels is not None and labels is not None:
                    aug_labels.append(labels[i])

        return aug_texts, aug_labels
