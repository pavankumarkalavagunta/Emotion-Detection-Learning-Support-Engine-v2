from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BILSTM_DIR, CHECKPOINTS_DIR, EMOTIONS, LABEL2ID, MAX_SEQ_LEN, ensure_project_dirs
from training.evaluate import save_evaluation_artifacts, save_training_curves
from training.preprocess import load_dataset


def build_bilstm(vocab_size: int, embedding_dim: int = 128) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=MAX_SEQ_LEN),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(EMOTIONS), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(csv_path: Path, epochs: int = 10, batch_size: int = 32, validation_size: float = 0.2) -> None:
    ensure_project_dirs()
    BILSTM_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=42,
        stratify=df["label"] if df["label"].value_counts().min() >= 2 else None,
    )

    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])
    x_train = pad_sequences(tokenizer.texts_to_sequences(train_df["clean_text"]), maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_df["clean_text"]), maxlen=MAX_SEQ_LEN, padding="post", truncating="post")
    y_train = to_categorical(train_df["label"], num_classes=len(EMOTIONS))
    y_val = to_categorical(val_df["label"], num_classes=len(EMOTIONS))

    model = build_bilstm(vocab_size=min(20000, len(tokenizer.word_index) + 1))
    checkpoint_path = CHECKPOINTS_DIR / "best_bilstm.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    model.save(BILSTM_DIR / "bilstm_model.keras")
    (BILSTM_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (BILSTM_DIR / "label_classes.json").write_text(json.dumps(EMOTIONS, indent=2), encoding="utf-8")
    (BILSTM_DIR / "label_mappings.json").write_text(json.dumps({"label2id": LABEL2ID}, indent=2), encoding="utf-8")

    y_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
    save_evaluation_artifacts(val_df["label"].to_numpy(), y_pred)
    save_training_curves(history.history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TensorFlow/Keras BiLSTM emotion classifier.")
    parser.add_argument("--csv", type=Path, default=Path("dataset/emotion_dataset.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()
    train(args.csv, args.epochs, args.batch_size, args.validation_size)


if __name__ == "__main__":
    main()
