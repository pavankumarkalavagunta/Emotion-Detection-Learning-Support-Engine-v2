from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATASET_DIR, EMOTIONS, LABEL2ID, ensure_project_dirs
from src.preprocessing import clean_text


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"text", "emotion"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns: {sorted(required)}. Missing: {sorted(missing)}")
    df = df[["text", "emotion"]].dropna()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.title()
    df = df[df["emotion"].isin(EMOTIONS)]
    if df.empty:
        raise ValueError(f"No rows matched supported emotions: {EMOTIONS}")
    df["clean_text"] = df["text"].apply(lambda value: clean_text(value, remove_stopwords=True, lemmatize=True))
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df["label"] = df["emotion"].map(LABEL2ID)
    return df


def preprocess_dataset(
    csv_path: Path,
    output_dir: Path = DATASET_DIR / "processed",
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, Path]:
    ensure_project_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(csv_path)
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    train_df, val_df = train_test_split(
        df,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validation.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    (output_dir / "label_mappings.json").write_text(
        json.dumps({"label2id": LABEL2ID, "id2label": {str(k): v for k, v in enumerate(EMOTIONS)}}, indent=2),
        encoding="utf-8",
    )
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a CSV emotion dataset.")
    parser.add_argument("--csv", type=Path, default=DATASET_DIR / "emotion_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / "processed")
    parser.add_argument("--validation-size", type=float, default=0.2)
    args = parser.parse_args()
    train_path, val_path = preprocess_dataset(args.csv, args.output_dir, args.validation_size)
    print(f"Saved train split to {train_path}")
    print(f"Saved validation split to {val_path}")


if __name__ == "__main__":
    main()
