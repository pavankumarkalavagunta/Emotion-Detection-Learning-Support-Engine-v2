from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
BILSTM_DIR = MODELS_DIR / "bilstm"
BERT_DIR = MODELS_DIR / "bert_emotion_model_final"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"
DB_PATH = DATA_DIR / "learning_assistant.db"
EXAMPLES_CSV = ROOT_DIR / "emotion_response_examples.csv"
MAPPING_CSV = ROOT_DIR / "emotion_response_mapping.csv"

EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Love", "Neutral"]
LABEL2ID = {label: index for index, label in enumerate(EMOTIONS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
ACADEMIC_FIELDS = [
    "Computer Science",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Engineering",
    "Business",
    "Literature",
    "History",
    "Psychology",
    "Other",
]

MAX_SEQ_LEN = 96
MIXED_EMOTION_THRESHOLD = 0.15


def ensure_project_dirs() -> None:
    for path in (DATA_DIR, DATASET_DIR, MODELS_DIR, BILSTM_DIR, BERT_DIR, CHECKPOINTS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
