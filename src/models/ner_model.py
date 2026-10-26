"""
Named Entity Recognition model — BIO token classification on code-mixed text.

Handles sub-word tokenization alignment: only the first sub-word piece
of each original token receives the true NER label.

Usage:
    from src.models.ner_model import NERModel
    model = NERModel(backbone="google/muril-base-cased", label_list=["O","B-PER",...])
    model.train(train_dataset, val_dataset)
    entities = model.predict("Shah Rukh Khan ki nayi movie release hui")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class NERModel:
    """Transformer-based NER model with BIO tagging.

    Parameters
    ----------
    backbone : str
        Hugging Face model name.
    label_list : list of str
        Ordered list of BIO labels (e.g. ["O", "B-PER", "I-PER", ...]).
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_length : int
        Maximum sequence length.
    """

    def __init__(
        self,
        backbone: str = "google/muril-base-cased",
        label_list: Optional[List[str]] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.backbone = backbone
        self.label_list = label_list or [
            "O", "B-PER", "I-PER", "B-LOC", "I-LOC",
            "B-ORG", "I-ORG", "B-MISC", "I-MISC",
        ]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        logger.info(f"Loading NER model: {backbone} ({self.num_labels} labels)")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForTokenClassification.from_pretrained(
            backbone,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)

    def _tokenize_and_align_labels(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenize pre-tokenised words and align NER labels to sub-words."""
        tokenized = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        all_labels = []
        for i, tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    tag = tags[word_idx] if word_idx < len(tags) else "O"
                    label_ids.append(self.label2id.get(tag, 0))
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        return tokenized

    def _compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Compute entity-level precision, recall, F1 via seqeval."""
        from seqeval.metrics import (
            f1_score,
            precision_score,
            recall_score,
        )

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Convert IDs back to label strings, ignoring -100
        true_labels = []
        pred_labels = []
        for pred_seq, label_seq in zip(preds, labels):
            true_seq = []
            pred_seq_str = []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    true_seq.append(self.id2label.get(l, "O"))
                    pred_seq_str.append(self.id2label.get(p, "O"))
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_str)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "models/ner",
        num_epochs: int = 15,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        **kwargs: Any,
    ) -> None:
        """Fine-tune the model for NER."""
        train_tokenized = train_dataset.map(
            self._tokenize_and_align_labels, batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_tokenized = None
        if val_dataset is not None:
            val_tokenized = val_dataset.map(
                self._tokenize_and_align_labels, batched=True,
                remove_columns=val_dataset.column_names,
            )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch" if val_tokenized else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if val_tokenized else False,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to="none",
            **kwargs,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting NER model training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Predict entities in a single text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list of dict
            Each dict has ``entity``, ``word``, ``start``, ``end``, ``score``.
        """
        from transformers import pipeline as hf_pipeline

        ner_pipeline = hf_pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",
        )
        results = ner_pipeline(text)

        entities = []
        for r in results:
            entities.append({
                "entity": r["entity_group"],
                "word": r["word"],
                "start": r["start"],
                "end": r["end"],
                "score": round(r["score"], 4),
            })
        return entities

    @classmethod
    def from_pretrained(
        cls, path: str, label_list: Optional[List[str]] = None, **kwargs: Any
    ) -> "NERModel":
        """Load a fine-tuned NER model from disk."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForTokenClassification.from_pretrained(path)
        instance.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        instance.id2label = instance.model.config.id2label
        instance.label2id = instance.model.config.label2id
        instance.label_list = label_list or list(instance.id2label.values())
        instance.num_labels = len(instance.label_list)
        instance.max_length = kwargs.get("max_length", 512)
        instance.backbone = path
        return instance
