from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.config import BERT_DIR, EMOTIONS, MAX_SEQ_LEN
from src.preprocessing import dominant_prediction


class BERTEmotionClassifier:
    def __init__(self, model_dir: Path = BERT_DIR) -> None:
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.id2label = {i: label for i, label in enumerate(EMOTIONS)}
        self.device = "cpu"
        self.loaded = False
        self.load_model()

    def load_model(self) -> None:
        if not self.model_dir.exists():
            return

        try:
            import torch
            from transformers import BertForSequenceClassification, BertTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()

            if getattr(self.model.config, "id2label", None):
                self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}
            self.loaded = True
        except Exception:
            self.tokenizer = None
            self.model = None
            self.loaded = False

    def predict(self, text: str) -> dict[str, Any]:
        if not self.loaded or self.tokenizer is None or self.model is None:
            return dominant_prediction(text, "BERT Fallback")

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_SEQ_LEN,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            scores = {self.id2label.get(i, EMOTIONS[i]): float(probs[i]) for i in range(len(probs))}
            emotion = max(scores, key=scores.get)
            return {
                "emotion": emotion,
                "confidence": float(scores[emotion]),
                "scores": scores,
                "cleaned_text": text.strip(),
                "model": "BERT",
            }
        except Exception:
            return dominant_prediction(text, "BERT Fallback")
