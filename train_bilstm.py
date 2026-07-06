"""
BiLSTM Model Training Script

This script trains a BiLSTM model for emotion detection using the dair-ai/emotion dataset.
It saves the trained model, tokenizer, and label encoder to the models/bilstm/ directory.

Usage:
    python train_bilstm.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from src.bilstm_model import BiLSTMEmotionModel
from src.config import (
    BATCH_SIZE,
    BILSTM_DIR,
    EMOTIONS,
    EPOCHS,
    MAX_SEQ_LEN,
    RANDOM_SEED,
    VOCAB_SIZE,
)
from src.preprocessing import TextPreprocessor


def load_and_prepare_data():
    """Load dair-ai/emotion dataset and prepare for training."""
    print("Loading dair-ai/emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")

    # Extract texts and labels
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    val_texts = dataset["validation"]["text"]
    val_labels = dataset["validation"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }


def preprocess_texts(texts, tokenizer=None, fit=False):
    """Clean and tokenize texts."""
    print("Preprocessing texts...")

    # Clean texts
    cleaned_texts = [TextPreprocessor.clean_text(text) for text in texts]

    # Fit tokenizer if needed
    if fit:
        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(cleaned_texts)

    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    return padded, tokenizer


def prepare_labels(labels):
    """Encode labels to one-hot vectors."""
    # Map dataset labels (0-5) to our emotion classes
    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTIONS)
    encoded = label_encoder.transform([EMOTIONS[i] for i in labels])
    categorical = to_categorical(encoded, num_classes=len(EMOTIONS))
    return categorical, label_encoder


def train_model():
    """Train BiLSTM model on emotion dataset."""
    print("\n" + "=" * 60)
    print("BiLSTM Emotion Detection Model Training")
    print("=" * 60 + "\n")

    # Set random seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load data
    data = load_and_prepare_data()

    # Preprocess training data
    print("\nPreprocessing training data...")
    train_padded, tokenizer = preprocess_texts(data["train_texts"], fit=True)
    val_padded, _ = preprocess_texts(data["val_texts"], tokenizer=tokenizer, fit=False)

    # Prepare labels
    print("Encoding labels...")
    train_labels, label_encoder = prepare_labels(data["train_labels"])
    val_labels, _ = prepare_labels(data["val_labels"])

    print(f"Training data shape: {train_padded.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Validation data shape: {val_padded.shape}\n")

    # Build and compile model
    print("Building BiLSTM model...")
    model = BiLSTMEmotionModel()
    model.summary()
    model.compile()

    # Train model
    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.model.fit(
        train_padded,
        train_labels,
        validation_data=(val_padded, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            ),
        ],
    )

    # Save model and related files
    print("\nSaving model and artifacts...")
    BILSTM_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(BILSTM_DIR)

    # Save tokenizer
    tokenizer_path = BILSTM_DIR / "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Save label encoder
    label_encoder_path = BILSTM_DIR / "label_encoder.pkl"
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {label_encoder_path}")

    # Save emotions list
    emotions_path = BILSTM_DIR / "emotions.json"
    with open(emotions_path, "w") as f:
        json.dump(EMOTIONS, f)
    print(f"Emotions saved to {emotions_path}")

    print("\n" + "=" * 60)
    print("✓ BiLSTM Model Training Complete!")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    model, history = train_model()
