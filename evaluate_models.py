"""
Model Evaluation Script

Evaluates both BiLSTM and BERT models on the emotion dataset and compares performance.

Usage:
    python evaluate_models.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.bilstm_model import BiLSTMPredictor
from src.bert_model import BERTEmotionClassifier
from src.config import BILSTM_DIR, BERT_DIR, EMOTIONS
from src.preprocessing import TextPreprocessor, preprocess_for_bilstm
import tensorflow as tf


def evaluate_bilstm(test_texts, test_labels, tokenizer):
    """Evaluate BiLSTM model."""
    print("\nEvaluating BiLSTM Model...")
    
    try:
        predictor = BiLSTMPredictor(BILSTM_DIR)
    except FileNotFoundError:
        print("✗ BiLSTM model not found. Please run train_bilstm.py first.")
        return None

    predictions = []
    for text in test_texts:
        result = predictor.predict(text)
        emotion_idx = EMOTIONS.index(result["emotion"])
        predictions.append(emotion_idx)

    predictions = np.array(predictions)
    test_labels = np.array(test_labels)

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")

    return {
        "model": "BiLSTM",
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predictions": predictions,
        "labels": test_labels,
    }


def evaluate_bert(test_texts, test_labels):
    """Evaluate BERT model."""
    print("\nEvaluating BERT Model...")
    
    try:
        predictor = BERTEmotionClassifier(BERT_DIR)
    except FileNotFoundError:
        print("✗ BERT model not found. Please run train_bert.py first.")
        return None

    predictions = []
    for text in test_texts:
        result = predictor.predict(text)
        emotion_idx = EMOTIONS.index(result["emotion"])
        predictions.append(emotion_idx)

    predictions = np.array(predictions)
    test_labels = np.array(test_labels)

    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="weighted")
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")

    return {
        "model": "BERT",
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "predictions": predictions,
        "labels": test_labels,
    }


def evaluate_models():
    """Evaluate both models and create comparison report."""
    print("\n" + "=" * 60)
    print("Emotion Detection Models Evaluation")
    print("=" * 60)

    # Load dataset
    print("\nLoading test dataset...")
    dataset = load_dataset("dair-ai/emotion")
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    # Load BiLSTM tokenizer for preprocessing
    try:
        with open(BILSTM_DIR / "tokenizer.pkl", "rb") as f:
            bilstm_tokenizer = pickle.load(f)
    except FileNotFoundError:
        bilstm_tokenizer = None

    # Evaluate models
    bilstm_results = evaluate_bilstm(test_texts, test_labels, bilstm_tokenizer)
    bert_results = evaluate_bert(test_texts, test_labels)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if bilstm_results:
        print(f"\nBiLSTM Model:")
        print(f"  Accuracy:  {bilstm_results['accuracy']:.4f}")
        print(f"  F1-Score:  {bilstm_results['f1']:.4f}")
        print(f"  Precision: {bilstm_results['precision']:.4f}")
        print(f"  Recall:    {bilstm_results['recall']:.4f}")

    if bert_results:
        print(f"\nBERT Model:")
        print(f"  Accuracy:  {bert_results['accuracy']:.4f}")
        print(f"  F1-Score:  {bert_results['f1']:.4f}")
        print(f"  Precision: {bert_results['precision']:.4f}")
        print(f"  Recall:    {bert_results['recall']:.4f}")

    # Detailed classification report
    if bilstm_results:
        print("\n" + "-" * 60)
        print("BiLSTM Classification Report:")
        print("-" * 60)
        print(classification_report(bilstm_results["labels"], bilstm_results["predictions"], target_names=EMOTIONS))

    if bert_results:
        print("-" * 60)
        print("BERT Classification Report:")
        print("-" * 60)
        print(classification_report(bert_results["labels"], bert_results["predictions"], target_names=EMOTIONS))

    # Save results
    results = {
        "bilstm": bilstm_results if bilstm_results else None,
        "bert": bert_results if bert_results else None,
    }

    results_path = Path("evaluation_results.json")
    with open(results_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_safe_results = {
            "bilstm": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in (bilstm_results.items() if bilstm_results else {}.items())
            },
            "bert": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in (bert_results.items() if bert_results else {}.items())
            },
        }
        json.dump(json_safe_results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    print("\n" + "=" * 60)
    print("✓ Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_models()
