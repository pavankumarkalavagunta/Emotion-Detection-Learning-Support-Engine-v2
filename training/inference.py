from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bert_model import BERTEmotionClassifier
from src.preprocessing import EMOTION_RESPONSES, EmotionPredictor


def predict_text(text: str, model_name: str = "bert") -> dict[str, Any]:
    start = time.perf_counter()
    if model_name.lower() == "bilstm":
        result = EmotionPredictor().predict(text)
    elif model_name.lower() == "compare":
        bilstm = EmotionPredictor().predict(text)
        bert = BERTEmotionClassifier().predict(text)
        return {
            "input_text": text,
            "prediction_time": time.perf_counter() - start,
            "results": {"bilstm": bilstm, "bert": bert},
        }
    else:
        result = BERTEmotionClassifier().predict(text)
    result["input_text"] = text
    result["prediction_time"] = time.perf_counter() - start
    result["recommendation"] = EMOTION_RESPONSES[result["emotion"]]["response"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run emotion inference from the command line.")
    parser.add_argument("text")
    parser.add_argument("--model", choices=["bert", "bilstm", "compare"], default="bert")
    args = parser.parse_args()
    print(predict_text(args.text, args.model))


if __name__ == "__main__":
    main()
