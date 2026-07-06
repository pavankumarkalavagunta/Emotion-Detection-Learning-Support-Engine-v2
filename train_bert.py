"""
BERT Fine-tuning Script

This script fine-tunes bert-base-uncased for emotion detection using the dair-ai/emotion dataset.
It saves the fine-tuned model with all necessary files to the models/bert_emotion_model_final/ directory.

Usage:
    python train_bert.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.config import BATCH_SIZE, BERT_DIR, BERT_MODEL_NAME, EMOTIONS, EPOCHS, RANDOM_SEED


def compute_metrics(eval_pred):
    """Compute accuracy and other metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def preprocess_function(examples, tokenizer):
    """Tokenize texts."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def train_bert_model():
    """Fine-tune BERT for emotion classification."""
    print("\n" + "=" * 60)
    print("BERT Fine-tuning for Emotion Detection")
    print("=" * 60 + "\n")

    # Set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load dataset
    print("Loading dair-ai/emotion dataset...")
    dataset = load_dataset("dair-ai/emotion")

    # Create label mapping
    label2id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
    id2label = {i: emotion for i, emotion in enumerate(EMOTIONS)}

    print(f"Dataset splits: {dataset.keys()}")
    print(f"Label mapping: {label2id}\n")

    # Load tokenizer and model
    print(f"Loading {BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=len(EMOTIONS),
        id2label=id2label,
        label2id=label2id,
    )

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenize_fn = lambda examples: preprocess_function(examples, tokenizer)
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Rename label column
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    print(f"Tokenized training set size: {len(tokenized_dataset['train'])}")
    print(f"Tokenized validation set size: {len(tokenized_dataset['validation'])}")
    print(f"Tokenized test set size: {len(tokenized_dataset['test'])}\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(BERT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        seed=RANDOM_SEED,
        logging_steps=100,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting fine-tuning training...\n")
    train_result = trainer.train()

    # Save model
    print("\nSaving fine-tuned model...")
    BERT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(BERT_DIR)

    # Save label mapping
    label_mapping = {"label2id": label2id, "id2label": id2label}
    with open(BERT_DIR / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("Test Results:")
    for key, val in test_results.items():
        print(f"  {key}: {val:.4f}")

    print("\n" + "=" * 60)
    print("✓ BERT Fine-tuning Complete!")
    print(f"Model saved to: {BERT_DIR}")
    print("=" * 60)

    return trainer, train_result, test_results


if __name__ == "__main__":
    trainer, train_result, test_results = train_bert_model()
