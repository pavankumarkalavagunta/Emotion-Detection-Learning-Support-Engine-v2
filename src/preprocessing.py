from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

import numpy as np

from src.config import BILSTM_DIR, EMOTIONS, MAX_SEQ_LEN

KEYWORDS: dict[str, list[str]] = {
    "Happy": ["happy", "glad", "excited", "joy", "great", "proud", "motivated", "confident"],
    "Sad": ["sad", "down", "upset", "hopeless", "discouraged", "cry", "unhappy"],
    "Angry": ["angry", "mad", "annoyed", "irritated", "frustrated", "hate", "furious"],
    "Fear": ["afraid", "fear", "scared", "worried", "anxious", "panic", "nervous"],
    "Surprise": ["surprised", "wow", "unexpected", "shocked", "amazed", "sudden"],
    "Love": ["love", "like", "enjoy", "passionate", "interested", "curious", "fascinated"],
    "Neutral": ["okay", "fine", "normal", "neutral", "average", "regular"],
}

EMOTION_RESPONSES: dict[str, dict[str, str]] = {
    "Happy": {"tone": "positive", "response": "You sound energized. Use that momentum to solve one slightly harder example and explain your reasoning out loud.", "action": "Advance to a stretch problem."},
    "Sad": {"tone": "low", "response": "This feels heavy right now. Start with a small, winnable step and let progress rebuild your confidence.", "action": "Review one simple example first."},
    "Angry": {"tone": "tense", "response": "Frustration is a sign the task needs to be reduced. Pause briefly, isolate the exact failing step, and retry with a worked solution nearby.", "action": "Debug the smallest failing part."},
    "Fear": {"tone": "anxious", "response": "Anxiety often shrinks when the task becomes predictable. Make a short checklist and complete the first item only.", "action": "Use a step-by-step checklist."},
    "Surprise": {"tone": "alert", "response": "That unexpected result is worth investigating. Compare it with what you expected, then write one rule that explains the difference.", "action": "Map expectation versus result."},
    "Love": {"tone": "engaged", "response": "Your interest is a strength. Follow it with a focused question and connect the answer back to the core concept.", "action": "Explore one deeper example."},
    "Neutral": {"tone": "steady", "response": "You seem steady. Keep the rhythm by practicing one representative problem, then check it against the solution.", "action": "Continue with guided practice."},
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}


def expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def clean_text(text: str, *, remove_stopwords: bool = False, lemmatize: bool = False) -> str:
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_stopwords or lemmatize:
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            for package, resource in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")):
                try:
                    nltk.data.find(resource)
                except LookupError:
                    nltk.download(package, quiet=True)
            tokens = text.split()
            if remove_stopwords:
                stop_words = set(stopwords.words("english"))
                tokens = [token for token in tokens if token not in stop_words]
            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            text = " ".join(tokens)
        except Exception:
            pass
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
        model_path = self.model_dir / "bilstm_model.keras"
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
            probs = probs / np.sum(probs) if np.sum(probs) else np.ones(len(self.classes)) / len(self.classes)
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
