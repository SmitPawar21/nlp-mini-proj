"""
Intent detection model — multi-class classification on code-mixed text.

Architecturally identical to the sentiment model but with a different
label set. Separated for clarity and independent configuration.

Usage:
    from src.models.intent_model import IntentModel
    model = IntentModel(backbone="google/muril-base-cased", num_labels=26)
    model.train(train_dataset, val_dataset)
    intent, score = model.predict("mujhe Delhi ka flight book karna hai")
"""

from __future__ import annotations

import json
import logging
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


class IntentModel:
    """Transformer-based intent classifier.

    Parameters
    ----------
    backbone : str
        Hugging Face model name.
    num_labels : int
        Number of intent classes (determined at runtime from dataset).
    label_map : dict, optional
        Mapping from label ID to intent name.
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_length : int
        Maximum sequence length.
    """

    def __init__(
        self,
        backbone: str = "google/muril-base-cased",
        num_labels: int = 26,
        label_map: Optional[Dict[int, str]] = None,
        device: Optional[str] = None,
        max_length: int = 128,  # Intents are typically short utterances
    ) -> None:
        self.backbone = backbone
        self.num_labels = num_labels
        self.label_map = label_map or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        logger.info(f"Loading intent model: {backbone} ({num_labels} labels)")
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
            padding=False,
        )

    def _compute_metrics(self, eval_pred: Any) -> Dict[str, float]:
        """Compute accuracy and macro-F1."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        }

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "models/intent",
        num_epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        freeze_backbone: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Fine-tune the model for intent detection."""
        from pathlib import Path

        if freeze_backbone is None:
            freeze_backbone = not torch.cuda.is_available()

        # Extract samples from Path, list, or HF Dataset
        if isinstance(train_dataset, (str, Path)):
            with open(train_dataset, "r", encoding="utf-8") as f:
                train_data = [json.loads(line) for line in f if line.strip()]
        elif isinstance(train_dataset, list):
            train_data = train_dataset
        elif hasattr(train_dataset, "to_list"):
            train_data = train_dataset.to_list()
        else:
            try:
                train_data = [dict(train_dataset[i]) for i in range(len(train_dataset))]
            except Exception:
                train_data = list(train_dataset)

        unique_labels = sorted(set(d.get("intent", d.get("label")) for d in train_data))
        self.label_map = {i: str(l) for i, l in enumerate(unique_labels)}
        label2id = {str(l): i for i, l in enumerate(unique_labels)}
        self.num_labels = len(unique_labels)

        logger.info(f"Detected {self.num_labels} intent classes, initializing model head")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.backbone, num_labels=self.num_labels
        )
        self.model.to(self.device)

        if freeze_backbone:
            base = getattr(self.model, "bert", getattr(self.model, "roberta", getattr(self.model, "base_model", None)))
            if base is not None:
                for param in base.parameters():
                    param.requires_grad = False
                logger.info("Froze backbone encoder; fine-tuning classification head.")

        texts = [d["text"] for d in train_data]
        labels = [label2id[str(d.get("intent", d.get("label")))] for d in train_data]

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            labels_tensor,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt_lr = 1e-3 if freeze_backbone else learning_rate
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=opt_lr)

        logger.info(f"Starting PyTorch training: {len(texts)} samples, {num_epochs} epochs, batch_size={batch_size}")
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for b_ids, b_mask, b_lbls in loader:
                optimizer.zero_grad()
                b_ids = b_ids.to(self.device)
                b_mask = b_mask.to(self.device)
                b_lbls = b_lbls.to(self.device)
                outputs = self.model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbls)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(loader))
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        label_map_path = Path(output_dir) / "label_map.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.label_map.items()}, f, indent=2)
        logger.info(f"Model + label_map saved to {output_dir}")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a single utterance.

        Returns
        -------
        tuple of (str, float)
            Predicted intent name and confidence score.
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

        intent_name = self.label_map.get(pred_id, str(pred_id))
        return intent_name, round(confidence, 4)

    def predict_with_top_k(
        self, text: str, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Return top-k intent predictions with scores."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        top_k = torch.topk(probs, k=min(k, self.num_labels))
        results = []
        for idx, score in zip(top_k.indices.tolist(), top_k.values.tolist()):
            intent_name = self.label_map.get(idx, str(idx))
            results.append((intent_name, round(score, 4)))
        return results

    @classmethod
    def from_pretrained(
        cls, path: str, num_labels: int = 26, **kwargs: Any
    ) -> "IntentModel":
        """Load a fine-tuned intent model from disk."""
        from pathlib import Path

        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        instance.max_length = kwargs.get("max_length", 128)
        instance.backbone = path

        # Load persisted label map
        label_map_path = Path(path) / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            instance.label_map = {int(k): v for k, v in raw.items()}
            instance.num_labels = len(instance.label_map)
        else:
            instance.label_map = kwargs.get("label_map", {})
            instance.num_labels = num_labels

        return instance
