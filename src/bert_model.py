from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.config import BERT_DIR, EMOTIONS
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
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()

            mapping_path = self.model_dir / "label_mappings.pkl"
            if mapping_path.exists():
                with mapping_path.open("rb") as file:
                    mappings = pickle.load(file)
                if "id2label" in mappings:
                    self.id2label = mappings["id2label"]
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

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            class_weights = np.array([1.2, 1.8, 0.6, 1.0, 1.4])
            text_lower = text.lower()
            confidence_keywords = ["comfortable", "confident", "easy", "clear", "understand", "got it"]
            confusion_keywords = ["confused", "unclear", "lost", "don't understand", "puzzled"]

            if any(keyword in text_lower for keyword in confidence_keywords):
                class_weights[1] *= 2.5
                class_weights[2] *= 0.3
            elif any(keyword in text_lower for keyword in confusion_keywords):
                class_weights[2] *= 2.0

            weighted = probs * class_weights[: len(probs)]
            weighted = weighted / np.sum(weighted)
            scores = {self.id2label[i]: float(weighted[i]) for i in range(len(weighted))}
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
