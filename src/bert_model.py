"""BERT Emotion Detection Model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import BERT_DIR, EMOTIONS


class BERTEmotionClassifier:
    """Load and predict with fine-tuned BERT model for emotion detection."""

    def __init__(self, model_dir: Path = BERT_DIR):
        """Initialize BERT classifier and load model."""
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        self.load_model()

    def load_model(self) -> None:
        """Load BERT model and tokenizer."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Please run train_bert.py first."
            )

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"✓ BERT model loaded from {self.model_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BERT model: {str(e)}")

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict emotion for input text using BERT.

        Args:
            text: Input text string

        Returns:
            Dictionary with emotion, confidence, and scores
        """
        if not self.loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Please run load_model() first.")

        from src.preprocessing import preprocess_for_bert

        try:
            # Preprocess text
            cleaned_text = preprocess_for_bert(text)

            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Get emotion and confidence
            emotion_idx = np.argmax(probs)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(probs[emotion_idx])

            # Create scores dictionary
            scores = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}

            return {
                "emotion": emotion,
                "confidence": confidence,
                "scores": scores,
                "model": "BERT",
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
