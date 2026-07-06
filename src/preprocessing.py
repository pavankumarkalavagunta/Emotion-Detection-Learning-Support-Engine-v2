from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import MAX_SEQ_LEN, VOCAB_SIZE, EMOTIONS


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline for emotion detection."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by:
        - Converting to lowercase
        - Removing URLs
        - Removing emojis
        - Removing special characters and extra whitespace
        - Removing punctuation

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = str(text).lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)

        # Remove emojis and special Unicode characters
        text = re.sub(
            r"[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]",
            "",
            text,
        )

        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", text).strip()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace again
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def tokenize_and_pad(
        texts: list[str],
        max_length: int = MAX_SEQ_LEN,
        vocab_size: int = VOCAB_SIZE,
        fit_on_texts: bool = True,
        tokenizer: Tokenizer | None = None,
    ) -> tuple[np.ndarray, Tokenizer]:
        """
        Tokenize texts and pad sequences.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            vocab_size: Maximum vocabulary size
            fit_on_texts: Whether to fit tokenizer on texts
            tokenizer: Pre-fitted tokenizer (if fit_on_texts=False)

        Returns:
            Tuple of (padded sequences array, tokenizer object)
        """
        if fit_on_texts:
            tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
            tokenizer.fit_on_texts(texts)
        elif tokenizer is None:
            raise ValueError("Must provide tokenizer if fit_on_texts=False")

        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

        return padded, tokenizer

    @staticmethod
    def preprocess_batch(texts: list[str]) -> np.ndarray:
        """
        Preprocess a batch of texts in one step.

        Args:
            texts: List of raw text strings

        Returns:
            Preprocessed and padded sequences
        """
        # Clean texts
        cleaned_texts = [TextPreprocessor.clean_text(text) for text in texts]

        # Tokenize and pad
        padded_sequences, _ = TextPreprocessor.tokenize_and_pad(
            cleaned_texts, fit_on_texts=True
        )

        return padded_sequences


def preprocess_for_bert(text: str) -> str:
    """
    Preprocess text for BERT model.
    BERT has its own tokenizer, so we only clean the text.

    Args:
        text: Raw text string

    Returns:
        Cleaned text suitable for BERT
    """
    return TextPreprocessor.clean_text(text)


def preprocess_for_bilstm(text: str, tokenizer: Tokenizer) -> np.ndarray:
    """
    Preprocess text for BiLSTM model.

    Args:
        text: Raw text string
        tokenizer: Fitted Keras tokenizer

    Returns:
        Padded sequence array
    """
    cleaned = TextPreprocessor.clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    return padded


# Emotion response mapping for study advice generation
EMOTION_RESPONSES: dict[str, dict[str, str]] = {
    "sadness": {
        "tone": "compassionate and supportive",
        "response": "It sounds like you're feeling down. Take a break if you need to, but remember that learning through challenges can be meaningful. Consider talking to someone or taking a short walk before returning.",
        "action": "Take a 15-minute break and return refreshed.",
    },
    "joy": {
        "tone": "enthusiastic and encouraging",
        "response": "That's wonderful! Your positive energy is fantastic for learning. Ride this momentum by tackling a slightly more challenging topic to deepen your understanding.",
        "action": "Move to a more advanced topic.",
    },
    "love": {
        "tone": "inspiring and passionate",
        "response": "Your passion for this topic is amazing! Channel that love into going deeper—explore related concepts, work on a project, or teach someone else what you've learned.",
        "action": "Create a project or teach the concept.",
    },
    "anger": {
        "tone": "calm and grounding",
        "response": "Frustration is a sign that something isn't clicking. Pause for a moment, take some deep breaths, and try breaking the problem into smaller, manageable steps.",
        "action": "Simplify the current problem.",
    },
    "fear": {
        "tone": "reassuring and gentle",
        "response": "Feeling anxious about this material is normal. You've learned difficult things before. Start with a review of fundamentals, then progress one small step at a time.",
        "action": "Review fundamentals first.",
    },
    "surprise": {
        "tone": "curious and exploratory",
        "response": "Something unexpected caught your attention! This is a perfect opportunity to explore further. Ask yourself what surprised you and see if you can understand why.",
        "action": "Investigate the surprising element.",
    },
    "neutral": {
        "tone": "professional and balanced",
        "response": "You seem to be in a neutral mindset. This is a good time to focus on steady, consistent progress. Set a specific learning goal for the next session.",
        "action": "Set a specific learning goal.",
    },
}


def get_mixed_emotions(scores: dict[str, float], threshold: float = 0.15) -> list[tuple[str, float]]:
    """
    Get multiple emotions if their scores exceed threshold.

    Args:
        scores: Dictionary of emotion scores
        threshold: Minimum score for secondary emotions

    Returns:
        List of (emotion, score) tuples sorted by score
    """
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not sorted_scores:
        return []
    mixed = [sorted_scores[0]]
    mixed.extend((emotion, score) for emotion, score in sorted_scores[1:] if score >= threshold)
    return mixed
