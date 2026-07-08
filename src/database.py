from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DB_PATH, ROOT_DIR


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # OneDrive-managed folders can reject SQLite journal writes in this sandbox.
    # Disable rollback journaling so local demos still persist records.
    conn.execute("PRAGMA journal_mode = OFF")
    return conn


def init_db() -> None:
    schema = (ROOT_DIR / "schema.sql").read_text(encoding="utf-8")
    with get_connection() as conn:
        conn.executescript(schema)


def upsert_user(email: str, name: str, role: str = "student") -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (email, name, role, login_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(email) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                login_count = users.login_count + 1
            """,
            (email, name, role),
        )


def insert_emotion_record(
    *,
    email: str,
    field: str,
    input_text: str,
    predicted_emotion: str,
    secondary_emotion: str | None,
    confidence_score: float,
    model_used: str,
    ai_response: str,
    response_type: str,
    emotion_scores: dict[str, float],
    csv_logged: bool,
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO emotion_records (
                email, field, input_text, predicted_emotion, secondary_emotion,
                confidence_score, model_used, ai_response, response_type,
                emotion_scores, timestamp, csv_logged
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email,
                field,
                input_text,
                predicted_emotion,
                secondary_emotion,
                confidence_score,
                model_used,
                ai_response,
                response_type,
                json.dumps(emotion_scores),
                datetime.now().isoformat(timespec="seconds"),
                int(csv_logged),
            ),
        )


def fetch_user_records(email: str | None = None) -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        if email:
            rows = conn.execute(
                "SELECT * FROM emotion_records WHERE email = ? ORDER BY timestamp DESC",
                (email,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM emotion_records ORDER BY timestamp DESC").fetchall()
    return [dict(row) for row in rows]
