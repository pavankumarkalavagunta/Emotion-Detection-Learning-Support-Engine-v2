from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
BILSTM_DIR = MODELS_DIR / "bltsm"
BERT_DIR = MODELS_DIR / "bert_emotion_model_final"
DB_PATH = DATA_DIR / "learning_assistant.db"
EXAMPLES_CSV = ROOT_DIR / "emotion_response_examples.csv"
MAPPING_CSV = ROOT_DIR / "emotion_response_mapping.csv"

EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
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

MAX_SEQ_LEN = 80
MIXED_EMOTION_THRESHOLD = 0.15
