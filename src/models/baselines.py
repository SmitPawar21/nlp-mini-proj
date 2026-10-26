"""
Classical ML baselines for code-mixed NLP tasks.

Provides:
- TF-IDF + Logistic Regression / SVM for text classification
  (Sentiment Analysis, Intent Detection)
- Character n-gram + CRF for sequence labelling (NER)

These serve as lower-bound comparisons for the Transformer models.

Usage:
    from src.models.baselines import TFIDFClassifier, CRFSequenceTagger
    clf = TFIDFClassifier(classifier_type="svm")
    clf.fit(train_texts, train_labels)
    preds = clf.predict(test_texts)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Classification Baseline
# ---------------------------------------------------------------------------

class TFIDFClassifier:
    """TF-IDF + classifier baseline for text classification.

    Parameters
    ----------
    classifier_type : str
        ``"lr"`` for Logistic Regression or ``"svm"`` for Linear SVM.
    max_features : int
        Maximum vocabulary size for TF-IDF.
    ngram_range : tuple of (int, int)
        N-gram range (e.g. (1, 2) for unigrams + bigrams).
    sublinear_tf : bool
        Apply sublinear TF scaling (log).
    max_iter : int
        Maximum iterations for the classifier.
    """

    def __init__(
        self,
        classifier_type: str = "lr",
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 3),
        sublinear_tf: bool = True,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.classifier_type = classifier_type

        # Build TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            analyzer="char_wb",  # Character n-grams work well for code-mixed text
            strip_accents=None,  # Preserve Devanagari
        )

        # Build classifier
        if classifier_type == "svm":
            clf = LinearSVC(
                max_iter=max_iter,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            clf = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                class_weight="balanced",
                solver="lbfgs",
            )

        self.pipeline = Pipeline([
            ("tfidf", tfidf),
            ("clf", clf),
        ])

    def save(self, filepath: str) -> None:
        """Save trained pipeline to disk."""
        import joblib

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"Saved TFIDFClassifier to {path}")

    @classmethod
    def load(cls, filepath: str) -> "TFIDFClassifier":
        """Load trained pipeline from disk."""
        import joblib

        instance = cls.__new__(cls)
        instance.pipeline = joblib.load(filepath)
        instance.classifier_type = "saved"
        return instance

    def fit(self, texts: List[str], labels: List[Any]) -> "TFIDFClassifier":
        """Fit the pipeline on training data."""
        logger.info(
            f"Training TF-IDF + {self.classifier_type.upper()} "
            f"on {len(texts)} examples..."
        )
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict labels for input texts."""
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probability distribution for input texts."""
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(texts)
        elif hasattr(clf, "decision_function"):
            df = self.pipeline.decision_function(texts)
            if df.ndim == 1:
                df = np.vstack([-df, df]).T
            exp = np.exp(df - np.max(df, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        else:
            raise NotImplementedError("Underlying classifier does not support probability estimation")

    def predict_with_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Predict top-k classes with confidence scores for a single text."""
        probs = self.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top_indices = np.argsort(probs)[::-1][:min(k, len(classes))]
        return [(str(classes[idx]), round(float(probs[idx]), 4)) for idx in top_indices]

    def evaluate(
        self,
        texts: List[str],
        labels: List[Any],
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Predict and compute classification metrics.

        Returns
        -------
        dict
            Contains ``report`` (str), ``confusion_matrix`` (ndarray),
            and ``predictions`` (ndarray).
        """
        preds = self.predict(texts)
        report = classification_report(
            labels, preds, target_names=label_names, digits=4
        )
        cm = confusion_matrix(labels, preds)

        logger.info(f"\n{report}")
        return {
            "report": report,
            "confusion_matrix": cm,
            "predictions": preds,
        }


# ---------------------------------------------------------------------------
# NER Baseline (CRF)
# ---------------------------------------------------------------------------

class CRFSequenceTagger:
    """CRF-based sequence tagger for NER baseline.

    Uses character n-gram and word-shape features.
    Requires ``python-crfsuite``.

    Parameters
    ----------
    c1 : float
        L1 regularization coefficient.
    c2 : float
        L2 regularization coefficient.
    max_iterations : int
        Maximum training iterations.
    """

    def __init__(
        self,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ) -> None:
        import pycrfsuite

        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.trainer.set_params({
            "c1": c1,
            "c2": c2,
            "max_iterations": max_iterations,
            "feature.possible_transitions": True,
        })
        self._tagger: Optional[Any] = None
        self._model_path = "crf_ner_baseline.model"

    @staticmethod
    def _word_features(tokens: List[str], idx: int) -> Dict[str, Any]:
        """Extract features for a single token in context.

        Features include:
        - Word shape (upper, lower, title, digit)
        - Character prefixes/suffixes (1-4 chars)
        - Context window (previous and next words)
        - Script type (Devanagari vs Latin)
        """
        word = tokens[idx]
        features: Dict[str, Any] = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word[-3:]": word[-3:],
            "word[-2:]": word[-2:],
            "word[:3]": word[:3],
            "word[:2]": word[:2],
            "word.isupper()": word.isupper(),
            "word.istitle()": word.istitle(),
            "word.isdigit()": word.isdigit(),
            "word.len": len(word),
            # Script detection (simple heuristic)
            "has_devanagari": any(
                0x0900 <= ord(c) <= 0x097F for c in word
            ),
            "all_latin": all(
                ord(c) < 128 for c in word if c.isalpha()
            ),
        }

        # Context features
        if idx > 0:
            prev = tokens[idx - 1]
            features.update({
                "-1:word.lower()": prev.lower(),
                "-1:word.istitle()": prev.istitle(),
                "-1:word.isupper()": prev.isupper(),
            })
        else:
            features["BOS"] = True  # Beginning of sentence

        if idx < len(tokens) - 1:
            nxt = tokens[idx + 1]
            features.update({
                "+1:word.lower()": nxt.lower(),
                "+1:word.istitle()": nxt.istitle(),
                "+1:word.isupper()": nxt.isupper(),
            })
        else:
            features["EOS"] = True  # End of sentence

        return features

    @classmethod
    def _extract_features(cls, tokens: List[str]) -> List[Dict[str, Any]]:
        """Extract features for all tokens in a sentence."""
        return [cls._word_features(tokens, i) for i in range(len(tokens))]

    @classmethod
    def load(cls, model_path: str) -> "CRFSequenceTagger":
        """Load trained CRF tagger from disk."""
        import pycrfsuite

        instance = cls.__new__(cls)
        instance._model_path = model_path
        instance._tagger = pycrfsuite.Tagger()
        instance._tagger.open(model_path)
        return instance

    def fit(
        self,
        tokens_list: List[List[str]],
        tags_list: List[List[str]],
        model_path: Optional[str] = None,
    ) -> "CRFSequenceTagger":
        """Train the CRF model.

        Parameters
        ----------
        tokens_list : list of list of str
            Training sentences (pre-tokenised).
        tags_list : list of list of str
            Corresponding NER tag sequences.
        """
        if model_path:
            self._model_path = model_path

        from pathlib import Path
        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training CRF on {len(tokens_list)} sentences...")

        for tokens, tags in zip(tokens_list, tags_list):
            features = self._extract_features(tokens)
            self.trainer.append(features, tags)

        self.trainer.train(self._model_path)
        logger.info(f"CRF model saved to {self._model_path}")

        # Load for prediction
        import pycrfsuite
        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(self._model_path)

        return self

    def predict(self, tokens_list: List[List[str]]) -> List[List[str]]:
        """Predict NER tags for a list of sentences."""
        if self._tagger is None:
            import pycrfsuite
            self._tagger = pycrfsuite.Tagger()
            self._tagger.open(self._model_path)

        predictions = []
        for tokens in tokens_list:
            features = self._extract_features(tokens)
            predicted_tags = self._tagger.tag(features)
            predictions.append(predicted_tags)
        return predictions

    def evaluate(
        self,
        tokens_list: List[List[str]],
        tags_list: List[List[str]],
    ) -> Dict[str, Any]:
        """Predict and compute entity-level metrics.

        Returns
        -------
        dict
            Contains ``report`` (str) and ``predictions``.
        """
        from seqeval.metrics import classification_report as seq_report

        preds = self.predict(tokens_list)
        report = seq_report(tags_list, preds, digits=4)
        logger.info(f"\n{report}")

        return {
            "report": report,
            "predictions": preds,
        }
