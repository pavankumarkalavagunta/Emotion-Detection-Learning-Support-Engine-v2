from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
BILSTM_DIR = MODELS_DIR / "bilstm"
BERT_DIR = MODELS_DIR / "bert_emotion_model_final"
DB_PATH = DATA_DIR / "learning_assistant.db"
EXAMPLES_CSV = ROOT_DIR / "emotion_response_examples.csv"
MAPPING_CSV = ROOT_DIR / "emotion_response_mapping.csv"

# Standard 7-emotion classification
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise", "neutral"]

# Human-friendly emotion names mapping
EMOTION_LABELS = {
    "sadness": "😢 Sad",
    "joy": "😊 Happy",
    "love": "❤️ Love",
    "anger": "😠 Angry",
    "fear": "😨 Fear",
    "surprise": "😲 Surprised",
    "neutral": "😐 Neutral"
}

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

# Model hyperparameters
MAX_SEQ_LEN = 128
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BATCH_SIZE = 32
EPOCHS = 3

# BERT model settings
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128

# Training settings
MIXED_EMOTION_THRESHOLD = 0.15
RANDOM_SEED = 42
