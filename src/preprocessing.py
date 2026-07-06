from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.config import BILSTM_DIR, EMOTIONS, MAX_SEQ_LEN

KEYWORDS: dict[str, list[str]] = {
    "Bored": [
        "bored",
        "boring",
        "tired",
        "repetitive",
        "dull",
        "not engaging",
        "too basic",
        "sleepy",
        "uninterested",
    ],
    "Confident": [
        "confident",
        "clear",
        "easy",
        "understand",
        "got it",
        "comfortable",
        "ready",
        "excellent",
        "sure",
    ],
    "Confused": [
        "confused",
        "unclear",
        "lost",
        "do not understand",
        "don't understand",
        "puzzled",
        "missing",
        "stuck",
    ],
    "Curious": [
        "curious",
        "why",
        "how",
        "what happens",
        "interested",
        "explore",
        "learn more",
        "fascinating",
    ],
    "Frustrated": [
        "frustrated",
        "frustrating",
        "annoying",
        "angry",
        "hate",
        "difficult",
        "wrong answer",
        "overwhelming",
    ],
}

EMOTION_RESPONSES: dict[str, dict[str, str]] = {
    "Bored": {
        "tone": "low energy",
        "response": "It makes sense that this feels dull or tiring. Try turning the topic into a short challenge, example, or real-world problem so your attention has something concrete to hold onto.",
        "action": "Use a short interactive exercise.",
    },
    "Confident": {
        "tone": "ready",
        "response": "Great progress. Since the idea feels clear, test your understanding with a harder question or explain the concept in your own words.",
        "action": "Move to a challenge problem.",
    },
    "Confused": {
        "tone": "uncertain",
        "response": "Confusion is a useful signal that one step needs to be unpacked. Start by naming the exact part that breaks down, then solve a smaller example before returning to the full problem.",
        "action": "Break the topic into smaller steps.",
    },
    "Curious": {
        "tone": "exploratory",
        "response": "That curiosity is valuable. Follow it by asking one focused question, then connect the answer back to the main concept you are studying.",
        "action": "Explore one deeper question.",
    },
    "Frustrated": {
        "tone": "stressed",
        "response": "Frustration usually means you have been pushing hard for a while. Pause, simplify the task, and retry one small step with a worked example beside you.",
        "action": "Switch to a simpler example first.",
    },
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keyword_scores(text: str) -> dict[str, float]:
    text_lower = clean_text(text)
    scores = {emotion: 0.2 for emotion in EMOTIONS}
    for emotion, keywords in KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[emotion] += 3.0 if keyword.lower() == emotion.lower() else 1.2
    total = sum(scores.values()) or 1.0
    return {emotion: score / total for emotion, score in scores.items()}


def dominant_prediction(text: str, model_name: str = "Keyword Fallback") -> dict[str, Any]:
    cleaned = clean_text(text)
    scores = keyword_scores(cleaned)
    emotion = max(scores, key=scores.get)
    return {
        "emotion": emotion,
        "confidence": float(scores[emotion]),
        "scores": scores,
        "cleaned_text": cleaned,
        "model": model_name,
    }


class EmotionPredictor:
    def __init__(self, model_dir: Path = BILSTM_DIR) -> None:
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.classes = EMOTIONS
        self.loaded = False
        self.load_model()

    def load_model(self) -> None:
        model_path = self.model_dir / "bilstm_student_adaptive.keras"
        tokenizer_path = self.model_dir / "tokenizer.json"
        classes_path = self.model_dir / "label_classes.json"
        if not model_path.exists() or not tokenizer_path.exists():
            return

        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import tokenizer_from_json

            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
            if classes_path.exists():
                self.classes = json.loads(classes_path.read_text(encoding="utf-8"))
            self.loaded = True
        except Exception:
            self.model = None
            self.tokenizer = None
            self.loaded = False

    def predict(self, text: str) -> dict[str, Any]:
        if not self.loaded or self.model is None or self.tokenizer is None:
            return dominant_prediction(text, "BiLSTM Fallback")

        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            cleaned = clean_text(text)
            sequence = self.tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
            probs = np.asarray(self.model.predict(padded, verbose=0)).reshape(-1)
            if len(probs) != len(self.classes):
                probs = np.resize(probs, len(self.classes))
            probs = np.exp(probs) / np.sum(np.exp(probs))
            scores = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
            emotion = max(scores, key=scores.get)
            return {
                "emotion": emotion,
                "confidence": float(scores[emotion]),
                "scores": scores,
                "cleaned_text": cleaned,
                "model": "BiLSTM",
            }
        except Exception:
            return dominant_prediction(text, "BiLSTM Fallback")


def get_mixed_emotions(scores: dict[str, float], threshold: float = 0.15) -> list[tuple[str, float]]:
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not sorted_scores:
        return []
    mixed = [sorted_scores[0]]
    mixed.extend((emotion, score) for emotion, score in sorted_scores[1:] if score >= threshold)
    return mixed
