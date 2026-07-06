# Emotion-Aware Learning Assistant

This project is a Streamlit-based AI learning assistant that detects learning-related emotions from text and returns supportive guidance. It includes a normalized database model with `Users` and `Emotion_Records`, CSV logging for analytics, optional Gemini responses, and model-loading hooks for BiLSTM and BERT assets.

## Features

- User profile capture with role support.
- Emotion prediction for Bored, Confident, Confused, Curious, and Frustrated.
- BiLSTM and BERT model integration with keyword fallback when model assets are absent.
- Mixed-emotion detection using a 15 percent secondary-emotion threshold.
- Gemini-powered learning guidance with template fallback.
- CSV and SQLite persistence.
- Streamlit analytics with emotion distribution, timeline, field breakdown, and summaries.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create your environment file.

```powershell
Copy-Item .env.example .env
```

Then add your Gemini key:

```text
GEMINI_API_KEY=your_key_here
```

4. Add trained model assets when available.

```text
models/
  bltsm/
    bilstm_student_adaptive.keras
    tokenizer.json
    label_classes.json
  bert_emotion_model_final/
    config.json
    model.safetensors or pytorch_model.bin
    tokenizer.json
    tokenizer_config.json
    vocab.txt
    label_mappings.pkl
```

The app runs without these files by using deterministic fallback prediction.

5. Start the app.

```powershell
streamlit run app.py
```

## Database Design

The schema is defined in `schema.sql`.

- `users.email` is the primary key.
- `emotion_records.record_id` is the primary key.
- `emotion_records.email` references `users.email`.
- One user can have many emotion analysis records.

The records table stores the field, input text, predicted and secondary emotions, confidence score, model used, AI response, response type, serialized emotion scores, timestamp, and CSV logging status.

## Project Flow

1. Environment setup and dependency configuration.
2. Kaggle model training and artifact export.
3. Local emotion detection pipeline integration.
4. AI-powered response generation and regeneration.
5. Streamlit UI and analytics dashboard.
6. End-to-end validation and deployment readiness.
