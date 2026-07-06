CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    password TEXT,
    role TEXT NOT NULL DEFAULT 'student',
    login_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS emotion_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    field TEXT NOT NULL,
    input_text TEXT NOT NULL,
    predicted_emotion TEXT NOT NULL,
    secondary_emotion TEXT,
    confidence_score REAL NOT NULL,
    model_used TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    response_type TEXT NOT NULL,
    emotion_scores TEXT NOT NULL,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    csv_logged INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_emotion_records_email
ON emotion_records(email);

CREATE INDEX IF NOT EXISTS idx_emotion_records_timestamp
ON emotion_records(timestamp);
