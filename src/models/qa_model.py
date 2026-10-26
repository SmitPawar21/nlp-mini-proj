"""
Question Answering model — extractive QA on code-mixed text.

Handles long contexts via sliding-window tokenization with stride.

Usage:
    from src.models.qa_model import QAModel
    model = QAModel(backbone="google/muril-base-cased")
    model.train(train_dataset, val_dataset)
    answer = model.predict(question="Shah Rukh Khan kahan rehte hain?",
                           context="Shah Rukh Khan Mumbai mein rehte hain...")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class QAModel:
    """Transformer-based extractive QA model.

    Parameters
    ----------
    backbone : str
        Hugging Face model name.
    device : str
        ``"cuda"`` or ``"cpu"``.
    max_length : int
        Maximum total sequence length (question + context).
    doc_stride : int
        Stride for sliding window on long contexts.
    max_answer_length : int
        Maximum answer span length.
    """

    def __init__(
        self,
        backbone: str = "google/muril-base-cased",
        device: Optional[str] = None,
        max_length: int = 384,
        doc_stride: int = 128,
        max_answer_length: int = 30,
    ) -> None:
        self.backbone = backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length

        logger.info(f"Loading QA model: {backbone}")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForQuestionAnswering.from_pretrained(backbone)
        self.model.to(self.device)

    def _prepare_train_features(self, examples: Dict[str, List]) -> Dict[str, Any]:
        """Tokenize QA examples and compute start/end positions."""
        tokenized = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answers = examples["answers"][sample_idx]

            # If no answer, set CLS position
            if not answers or not answers.get("answer_start"):
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find token range that contains the answer
            sequence_ids = tokenized.sequence_ids(i)

            # Find context start/end in token sequence
            ctx_start = 0
            while ctx_start < len(sequence_ids) and sequence_ids[ctx_start] != 1:
                ctx_start += 1
            ctx_end = len(sequence_ids) - 1
            while ctx_end >= 0 and sequence_ids[ctx_end] != 1:
                ctx_end -= 1

            # Check if answer is in this window
            if (
                offsets[ctx_start][0] > end_char
                or offsets[ctx_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Find start token
                tok_start = ctx_start
                while tok_start <= ctx_end and offsets[tok_start][0] <= start_char:
                    tok_start += 1
                start_positions.append(tok_start - 1)

                # Find end token
                tok_end = ctx_end
                while tok_end >= ctx_start and offsets[tok_end][1] >= end_char:
                    tok_end -= 1
                end_positions.append(tok_end + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    def train(
        self,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "models/qa",
        num_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        **kwargs: Any,
    ) -> None:
        """Fine-tune the model for extractive QA."""
        train_tokenized = train_dataset.map(
            self._prepare_train_features,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        val_tokenized = None
        if val_dataset is not None:
            val_tokenized = val_dataset.map(
                self._prepare_train_features,
                batched=True,
                remove_columns=val_dataset.column_names,
            )

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
            data_collator=DefaultDataCollator(),
            processing_class=self.tokenizer,
        )

        logger.info("Starting QA model training...")
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def predict(
        self,
        question: str,
        context: str,
        n_best: int = 5,
    ) -> Dict[str, Any]:
        """Extract the answer span from context given a question.

        Returns
        -------
        dict
            ``answer``, ``score``, ``start``, ``end``, and ``n_best`` candidates.
        """
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        sequence_ids = inputs.sequence_ids(0)
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()

        # Mask out non-context tokens (question, special tokens)
        for i, seq_id in enumerate(sequence_ids):
            if seq_id != 1:
                start_logits[i] = -10000.0
                end_logits[i] = -10000.0

        # Get top-n start and end positions within context
        start_idxs = np.argsort(start_logits)[::-1][:n_best]
        end_idxs = np.argsort(end_logits)[::-1][:n_best]

        # Find valid spans
        candidates = []
        for s in start_idxs:
            if start_logits[s] < -100.0:
                continue
            for e in end_idxs:
                if end_logits[e] < -100.0:
                    continue
                if e >= s and (e - s + 1) <= self.max_answer_length:
                    raw_score = float(start_logits[s] + end_logits[e])
                    start_char = int(offset_mapping[s][0].item())
                    end_char = int(offset_mapping[e][1].item())
                    answer = context[start_char:end_char].strip(" \t\n.,!?;:\"'()[]{}")
                    if answer:
                        # Calibrated probability using logistic sigmoid
                        prob = float(1.0 / (1.0 + np.exp(-raw_score / 2.5)))
                        prob = float(np.clip(prob, 0.05, 0.98))
                        candidates.append({
                            "answer": answer,
                            "score": round(prob, 4),
                            "raw_score": round(raw_score, 4),
                            "start": start_char,
                            "end": end_char,
                        })

        # Sort by score
        candidates.sort(key=lambda x: x["raw_score"], reverse=True)

        if candidates:
            return {
                "answer": candidates[0]["answer"],
                "score": candidates[0]["score"],
                "start": candidates[0]["start"],
                "end": candidates[0]["end"],
                "n_best": candidates[:n_best],
            }
        return {
            "answer": "",
            "score": 0.0,
            "start": 0,
            "end": 0,
            "n_best": [],
        }

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "QAModel":
        """Load a fine-tuned QA model from disk."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForQuestionAnswering.from_pretrained(path)
        instance.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        instance.model.to(instance.device)
        instance.max_length = kwargs.get("max_length", 384)
        instance.doc_stride = kwargs.get("doc_stride", 128)
        instance.max_answer_length = kwargs.get("max_answer_length", 30)
        instance.backbone = path
        return instance
