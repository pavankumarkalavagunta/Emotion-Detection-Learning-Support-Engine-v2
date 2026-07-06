# Project Analysis Report

## Overview

The Emotion-Aware Learning Assistant helps learners identify their emotional state while studying and receive targeted support. The system accepts a field of study and a problem description, predicts learning emotions, generates supportive responses, and stores interaction history for analytics.

## Database Model

The database contains two primary entities: `Users` and `Emotion_Records`. `Users` stores account metadata such as email, name, role, login count, and creation timestamp. `Emotion_Records` stores each analysis session, including input text, prediction results, confidence scores, AI response, model source, timestamp, and CSV logging status.

The relationship is one-to-many: one user can create many emotion records, while each emotion record belongs to exactly one user. This structure minimizes redundancy and supports secure, queryable interaction history.

## Implemented Epics

### Epic 1: Environment Setup

The project includes a Python dependency file, environment template, model folder expectations, and a Streamlit entry point. The app reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment.

### Epic 2: Model Training and Integration

The code expects Kaggle-trained BiLSTM assets under `models/bltsm/` and BERT assets under `models/bert_emotion_model_final/`. If those assets are missing, the app uses keyword-based fallback predictors so the interface and persistence layers remain testable.

### Epic 3: Emotion Detection Pipeline

The pipeline cleans input text, applies model prediction, returns a unified schema, detects mixed emotions above a 15 percent threshold, and exposes scores for visualization.

### Epic 4: AI Guidance Engine

The system builds a prompt using the selected academic field, detected emotion, confidence score, and student problem. Gemini is used when configured; otherwise, predefined template responses provide reliable support.

### Epic 5: Streamlit UI

The UI includes profile capture, field selection, text input, settings, prediction comparison, AI response display, session history, and analytics tabs.

### Epic 6: Validation and Deployment Readiness

The implementation supports local startup, fallback inference, CSV persistence, SQLite persistence, and analytics rendering. Full trained-model validation should be performed after copying exported model assets into the expected folders.

## Future Scope

Future extensions can add authentication, instructor dashboards, LMS integration, multilingual emotion detection, stronger privacy controls, cloud database deployment, and continuous model improvement from reviewed session data.
