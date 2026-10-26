"""
Tokenizer utilities for multilingual / code-mixed models.

Wraps Hugging Face tokenizers with convenience methods for:
- Loading the correct tokenizer for a model backbone
- Tokenizing for different task types (classification, token classification,
  seq2seq, QA)
- Handling the nuances of sub-word tokenization for NER label alignment

Usage:
    from src.preprocessing.tokenizer_utils import get_tokenizer, tokenize_for_task
    tokenizer = get_tokenizer("google/muril-base-cased")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(
    model_name: str,
    use_fast: bool = True,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """Load a Hugging Face tokenizer by model name.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier (e.g. "google/muril-base-cased").
    use_fast : bool
        Whether to use the Rust-based fast tokenizer.

    Returns
    -------
    PreTrainedTokenizerBase
    """
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, **kwargs)


def tokenize_for_classification(
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int = 512,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Tokenize a list of texts for sequence classification.

    Returns
    -------
    dict
        BatchEncoding with ``input_ids``, ``attention_mask``, etc.
    """
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        **kwargs,
    )


def tokenize_for_ner(
    tokenizer: PreTrainedTokenizerBase,
    tokens_list: List[List[str]],
    tags_list: List[List[int]],
    max_length: int = 512,
    label_all_tokens: bool = False,
) -> Dict[str, Any]:
    """Tokenize pre-tokenised words for token classification (NER).

    Handles sub-word alignment: only the first sub-word of each original
    token receives the true label; continuation sub-words receive -100.

    Parameters
    ----------
    tokens_list : list of list of str
        Pre-tokenised sentences.
    tags_list : list of list of int
        Corresponding NER tag IDs.
    max_length : int
        Maximum sequence length.
    label_all_tokens : bool
        If True, assign the same label to all sub-words of a token.

    Returns
    -------
    dict
        Contains ``input_ids``, ``attention_mask``, ``labels``.
    """
    tokenized = tokenizer(
        tokens_list,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    all_labels = []
    for i, tags in enumerate(tags_list):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First sub-word of a new token
                label_ids.append(tags[word_idx] if word_idx < len(tags) else -100)
            else:
                # Continuation sub-word
                if label_all_tokens:
                    label_ids.append(tags[word_idx] if word_idx < len(tags) else -100)
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def tokenize_for_qa(
    tokenizer: PreTrainedTokenizerBase,
    questions: List[str],
    contexts: List[str],
    max_length: int = 384,
    doc_stride: int = 128,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Tokenize question-context pairs for extractive QA.

    Uses a sliding window with *doc_stride* for long contexts.

    Returns
    -------
    dict
        BatchEncoding with overflow mapping for long documents.
    """
    return tokenizer(
        questions,
        contexts,
        padding=True,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        **kwargs,
    )


def tokenize_for_seq2seq(
    tokenizer: PreTrainedTokenizerBase,
    source_texts: List[str],
    target_texts: Optional[List[str]] = None,
    max_source_length: int = 256,
    max_target_length: int = 256,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Tokenize source and optional target texts for seq2seq models.

    Parameters
    ----------
    source_texts : list of str
        Input sequences (e.g. code-mixed text).
    target_texts : list of str, optional
        Target sequences (e.g. Hindi translation). If provided, tokenized
        as ``labels``.

    Returns
    -------
    dict
        Contains ``input_ids``, ``attention_mask``, and optionally ``labels``.
    """
    model_inputs = tokenizer(
        source_texts,
        padding=True,
        truncation=True,
        max_length=max_source_length,
        return_tensors="pt",
        **kwargs,
    )

    if target_texts is not None:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=max_target_length,
                return_tensors="pt",
            )
        model_inputs["labels"] = labels["input_ids"]

    return model_inputs
