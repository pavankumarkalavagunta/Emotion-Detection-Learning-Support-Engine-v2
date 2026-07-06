"""BiLSTM Emotion Detection Model."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import BILSTM_DIR, EMBEDDING_DIM, EMOTIONS, LSTM_UNITS, MAX_SEQ_LEN, VOCAB_SIZE


class BiLSTMEmotionModel:
    """BiLSTM model for emotion detection."""

    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM):
        """Initialize BiLSTM model architecture."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = LSTM_UNITS
        self.num_classes = len(EMOTIONS)
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build BiLSTM model architecture."""
        model = models.Sequential(
            [
                layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                    input_length=MAX_SEQ_LEN,
                    name="embedding",
                ),
                layers.Bidirectional(
                    layers.LSTM(self.lstm_units, return_sequences=True, dropout=0.3),
                    name="bilstm_1",
                ),
                layers.Bidirectional(
                    layers.LSTM(self.lstm_units // 2, return_sequences=False, dropout=0.3),
                    name="bilstm_2",
                ),
                layers.Dense(128, activation="relu", name="dense_1"),
                layers.Dropout(0.4, name="dropout_1"),
                layers.Dense(64, activation="relu", name="dense_2"),
                layers.Dropout(0.3, name="dropout_2"),
                layers.Dense(self.num_classes, activation="softmax", name="output"),
            ]
        )
        return model

    def compile(self) -> None:
        """Compile model with appropriate loss and metrics."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def summary(self) -> None:
        """Print model summary."""
        self.model.summary()

    def save_model(self, save_dir: Path = BILSTM_DIR) -> None:
        """Save trained model and associated files."""
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "bilstm_model.keras"
        self.model.save(model_path)
        print(f"Model saved to {model_path}")


class BiLSTMPredictor:
    """Load and predict with trained BiLSTM model."""

    def __init__(self, model_dir: Path = BILSTM_DIR):
        """Initialize predictor and load model."""
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.load_model()

    def load_model(self) -> None:
        """Load BiLSTM model and tokenizer."""
        model_path = self.model_dir / "bilstm_model.keras"
        tokenizer_path = self.model_dir / "tokenizer.pkl"

        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Model or tokenizer not found in {self.model_dir}. "
                "Please run train_bilstm.py first."
            )

        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
            self.loaded = True
            print(f"✓ BiLSTM model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BiLSTM model: {str(e)}")

    def predict(self, text: str) -> dict[str, Any]:
        """
        Predict emotion for input text.

        Args:
            text: Input text string

        Returns:
            Dictionary with emotion, confidence, and scores
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Please run load_model() first.")

        from src.preprocessing import preprocess_for_bilstm

        try:
            # Preprocess text
            padded = preprocess_for_bilstm(text, self.tokenizer)

            # Get predictions
            predictions = self.model.predict(padded, verbose=0)
            probs = predictions[0]

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
                "model": "BiLSTM",
            }
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
