"""
Sentiment analysis model — sequence classification on code-mixed text.

Supports multiple Transformer backbones (mBERT, XLM-R, MuRIL) configured
via YAML. The same architecture is reusable for any text classification
task by changing the label set.

Usage:
    from src.models.sentiment_model import SentimentModel
    model = SentimentModel(backbone="google/muril-base-cased", num_labels=3)
    model.train(train_dataset, val_dataset)
    label, confidence = model.predict("Yeh movie bahut acha hai")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class SentimentModel:
    """Transformer-based sentiment classifier.

    Parameters
    ----------
    backbone : str
        Hugging Face model name.
    num_labels : int
        Number of sentiment classes.
    label_map : dict, optional
        Mapping from label ID to label name.
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_length : int
        Maximum sequence length.
    """

    def __init__(
        self,
        backbone: str = "google/muril-base-cased",
        num_labels: int = 3,
        label_map: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
        max_length: int = 128,
    ) -> None:
        self.backbone = backbone
        self.num_labels = num_labels
        self.label_map = label_map or {0: "negative", 1: "neutral", 2: "positive"}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        logger.info(f"Loading sentiment model: {backbone} ({num_labels} labels)")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=num_labels
        )
        self.model.to(self.device)

    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenize texts for classification."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding via data collator
        )

    def _compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Compute accuracy, precision, recall, F1 for evaluation."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "models/sentiment",
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        **kwargs: Any,
    ) -> None:
        """Fine-tune the model for sentiment classification."""
        # Tokenize
        train_tokenized = train_dataset.map(
            self._tokenize_function, batched=True,
            remove_columns=[c for c in train_dataset.column_names if c != "label"],
        )
        val_tokenized = None
        if val_dataset is not None:
            val_tokenized = val_dataset.map(
                self._tokenize_function, batched=True,
                remove_columns=[c for c in val_dataset.column_names if c != "label"],
            )

        data_collator = DataCollatorWithPadding(self.tokenizer)

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
            logging_steps=50,
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

        logger.info("Starting sentiment model training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Persist label map
        label_map_path = Path(output_dir) / "label_map.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.label_map.items()}, f, indent=2)
        logger.info(f"Model saved to {output_dir}")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a single text.

        Returns
        -------
        tuple of (str, float)
            Predicted label name and confidence score.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        label_name = self.label_map.get(pred_id, str(pred_id))
        return label_name, round(confidence, 4)

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for a batch of texts."""
        return [self.predict(t) for t in texts]

    @classmethod
    def from_pretrained(
        cls, path: str, num_labels: int = 3, **kwargs: Any
    ) -> "SentimentModel":
        """Load a fine-tuned model from disk."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        instance.num_labels = num_labels
        
        # Load label map if available
        label_map_path = Path(path) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
            instance.label_map = {int(k): v for k, v in raw_map.items()}
        else:
            instance.label_map = kwargs.get("label_map", {0: "negative", 1: "neutral", 2: "positive"})

        instance.max_length = kwargs.get("max_length", 128)
        instance.backbone = path
        return instance
