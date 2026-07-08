from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BERT_DIR, CHECKPOINTS_DIR, EMOTIONS, ID2LABEL, LABEL2ID, MAX_SEQ_LEN, ensure_project_dirs
from training.evaluate import save_evaluation_artifacts
from training.preprocess import load_dataset


class EmotionDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: BertTokenizer) -> None:
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_SEQ_LEN)
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(eval_pred) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train(csv_path: Path, epochs: int = 3, batch_size: int = 8, validation_size: float = 0.2) -> None:
    ensure_project_dirs()
    df = load_dataset(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=42,
        stratify=df["label"] if df["label"].value_counts().min() >= 2 else None,
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = EmotionDataset(train_df["clean_text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_dataset = EmotionDataset(val_df["clean_text"].tolist(), val_df["label"].tolist(), tokenizer)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(EMOTIONS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR / "bert"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(CHECKPOINTS_DIR / "bert_logs"),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    save_evaluation_artifacts(val_df["label"].to_numpy(), y_pred)
    BERT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(BERT_DIR)
    tokenizer.save_pretrained(BERT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for emotion classification.")
    parser.add_argument("--csv", type=Path, default=Path("dataset/emotion_dataset.csv"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()
    train(args.csv, args.epochs, args.batch_size, args.validation_size)


if __name__ == "__main__":
    main()
