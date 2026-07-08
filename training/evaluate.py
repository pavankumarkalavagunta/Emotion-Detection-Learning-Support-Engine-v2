from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import EMOTIONS, REPORTS_DIR


def save_training_curves(history: dict[str, Iterable[float]], output_dir: Path = REPORTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "accuracy" in history:
        plt.figure()
        plt.plot(history.get("accuracy", []), label="train")
        plt.plot(history.get("val_accuracy", []), label="validation")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_accuracy.png")
        plt.close()
    if "loss" in history:
        plt.figure()
        plt.plot(history.get("loss", []), label="train")
        plt.plot(history.get("val_loss", []), label="validation")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png")
        plt.close()


def save_evaluation_artifacts(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    labels: list[str] = EMOTIONS,
    output_dir: Path = REPORTS_DIR,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(labels)))
    report = classification_report(y_true, y_pred, labels=label_ids, target_names=labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
