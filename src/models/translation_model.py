"""
Translation model — seq2seq fine-tuning for code-mixed → Hindi translation.

Wraps Hugging Face ``AutoModelForSeq2SeqLM`` for models like mT5,
IndicTrans2, or NLLB.

Usage:
    from src.models.translation_model import TranslationModel
    model = TranslationModel(backbone="google/mt5-small")
    model.train(train_dataset, val_dataset)
    output = model.translate("Yeh bahut acha movie hai")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)


class TranslationModel:
    """Seq2seq translation model for code-mixed text.

    Parameters
    ----------
    backbone : str
        Hugging Face model name (e.g. ``"google/mt5-small"``).
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_source_length : int
        Max tokens for source (code-mixed) text.
    max_target_length : int
        Max tokens for target (Hindi) text.
    """

    def __init__(
        self,
        backbone: str = "google/mt5-small",
        device: Optional[str] = None,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ) -> None:
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        logger.info(f"Loading translation model: {backbone}")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
        self.model.to(self.device)

    def _preprocess_function(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenize source and target for training."""
        # Add task prefix for mT5
        prefix = "translate code-mixed to Hindi: "
        inputs = [prefix + src for src in examples["source"]]
        targets = examples["target"]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
            )

        # Replace pad token id with -100 so they are ignored in loss
        labels["input_ids"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "models/translation",
        num_epochs: int = 15,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        **kwargs: Any,
    ) -> None:
        """Fine-tune the seq2seq model.

        Parameters
        ----------
        train_dataset : Dataset
            Hugging Face Dataset with ``source`` and ``target`` columns.
        val_dataset : Dataset, optional
            Validation dataset.
        """
        # Tokenize datasets
        train_tokenized = train_dataset.map(
            self._preprocess_function, batched=True, remove_columns=train_dataset.column_names
        )
        val_tokenized = None
        if val_dataset is not None:
            val_tokenized = val_dataset.map(
                self._preprocess_function, batched=True, remove_columns=val_dataset.column_names
            )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding=True
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            predict_with_generate=True,
            eval_strategy="epoch" if val_tokenized else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if val_tokenized else False,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to="none",
            **kwargs,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )

        logger.info("Starting translation model training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def translate(
        self,
        text: str,
        num_beams: int = 4,
        max_length: int = 256,
    ) -> str:
        """Translate a single code-mixed text to Hindi.

        Parameters
        ----------
        text : str
            Code-mixed input text.

        Returns
        -------
        str
            Hindi translation.
        """
        prefix = "translate code-mixed to Hindi: "
        inputs = self.tokenizer(
            prefix + text,
            return_tensors="pt",
            max_length=self.max_source_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(
        self, texts: List[str], batch_size: int = 16, **kwargs: Any
    ) -> List[str]:
        """Translate a batch of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.translate(text, **kwargs))
        return results

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "TranslationModel":
        """Load a fine-tuned model from disk."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        instance.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        instance.max_source_length = kwargs.get("max_source_length", 256)
        instance.max_target_length = kwargs.get("max_target_length", 256)
        instance.backbone = path
        return instance
