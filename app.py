from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.bert_model import BERTEmotionClassifier
from src.config import ACADEMIC_FIELDS, EXAMPLES_CSV, MAPPING_CSV, MIXED_EMOTION_THRESHOLD
from src.database import insert_emotion_record, upsert_user
from src.preprocessing import EMOTION_RESPONSES, EmotionPredictor, get_mixed_emotions

load_dotenv()


st.set_page_config(
    page_title="Emotion-Aware Learning Assistant",
    page_icon="ED",
    layout="wide",
)


@st.cache_resource
def load_models() -> tuple[EmotionPredictor, BERTEmotionClassifier, str]:
    bilstm = EmotionPredictor()
    bert = BERTEmotionClassifier()
    loaded = []
    loaded.append("BiLSTM" if bilstm.loaded else "BiLSTM fallback")
    loaded.append("BERT" if bert.loaded else "BERT fallback")
    return bilstm, bert, ", ".join(loaded)


@st.cache_data
def load_examples() -> pd.DataFrame:
    if EXAMPLES_CSV.exists():
        return pd.read_csv(EXAMPLES_CSV)
    return pd.DataFrame(columns=["text", "emotion", "confidence", "response", "field", "timestamp"])


def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        return genai.Client(api_key=api_key)
    except Exception:
        return None


def get_ai_response(field: str, problem: str, emotion: str, confidence: float) -> tuple[str, str]:
    model = configure_gemini()
    fallback = EMOTION_RESPONSES[emotion]["response"]
    if model is None:
        return fallback, "Template"

    prompt = f"""
You are a supportive learning assistant.
The learner studies {field}.
They feel {emotion} with {confidence:.1%} model confidence.
Problem: {problem}

Write a concise response with:
1. Acknowledgment of the emotion.
2. One field-specific learning strategy.
3. One encouraging next step.
Use simple language and no markdown.
"""
    try:
        response = model.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip(), "Gemini AI"
    except Exception:
        return fallback, "Template"


def save_to_csv(field: str, problem: str, emotion: str, confidence: float, response: str) -> bool:
    try:
        row = {
            "text": problem,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "response": response,
            "field": field,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if EXAMPLES_CSV.exists():
            examples = pd.read_csv(EXAMPLES_CSV)
            examples = pd.concat([examples, pd.DataFrame([row])], ignore_index=True)
        else:
            examples = pd.DataFrame([row])
        examples.to_csv(EXAMPLES_CSV, index=False)

        mapping_row = {"emotion": emotion, "response": response}
        if MAPPING_CSV.exists():
            mapping = pd.read_csv(MAPPING_CSV)
            if emotion not in set(mapping["emotion"].astype(str)):
                mapping = pd.concat([mapping, pd.DataFrame([mapping_row])], ignore_index=True)
        else:
            mapping = pd.DataFrame([mapping_row])
        mapping.to_csv(MAPPING_CSV, index=False)
        load_examples.clear()
        return True
    except Exception as exc:
        st.error(f"Failed to save CSV data: {exc}")
        return False


def add_history(
    field: str,
    problem: str,
    result: dict,
    ai_response: str,
    response_type: str,
) -> None:
    mixed = get_mixed_emotions(result["scores"], MIXED_EMOTION_THRESHOLD)
    emotion_label = " + ".join(item[0] for item in mixed) if len(mixed) > 1 else result["emotion"]
    secondary = mixed[1][0] if len(mixed) > 1 else None
    entry = {
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": emotion_label,
        "primary_emotion": result["emotion"],
        "secondary_emotion": secondary,
        "confidence": result["confidence"],
        "ai_response": ai_response,
        "all_scores": result["scores"],
        "model": result["model"],
        "response_type": response_type,
    }
    st.session_state.emotion_history.append(entry)


def show_prediction(title: str, result: dict) -> None:
    st.write(f"**{title}**")
    mixed = get_mixed_emotions(result["scores"], MIXED_EMOTION_THRESHOLD)
    if len(mixed) > 1:
        st.metric("Mixed Emotions", " + ".join(item[0] for item in mixed), f"Primary: {mixed[0][1]:.1%}")
    else:
        st.metric("Emotion", result["emotion"], f"{result['confidence']:.1%}")

    for emotion, score in sorted(result["scores"].items(), key=lambda item: item[1], reverse=True):
        st.progress(float(score), text=f"{emotion}: {score:.1%}")


def show_analytics() -> None:
    history = st.session_state.emotion_history
    if not history:
        return

    st.divider()
    st.header("Learning Analytics")
    df = pd.DataFrame(history)
    df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")

    tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["emotion"].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, title="Emotion Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(df, x="time", y="confidence", color="emotion", markers=True, title="Emotional Journey")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        grouped = df.groupby(["field", "emotion", "model"]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x="field",
            y="count",
            color="emotion",
            facet_col="model",
            title="Emotions by Study Field and Model",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2, col3 = st.columns(3)
        col1.metric("Interactions", len(df))
        col2.metric("Average Confidence", f"{df['confidence'].mean():.1%}")
        col3.metric("Most Common Emotion", df["emotion"].mode().iloc[0])
        st.dataframe(
            df[["timestamp", "field", "emotion", "confidence", "model", "response_type"]],
            use_container_width=True,
        )


def main() -> None:
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []

    bilstm_model, bert_model, status = load_models()
    examples_df = load_examples()

    with st.sidebar:
        st.header("Dashboard")
        st.write(f"Models: {status}")
        st.write(f"Total interactions: {len(st.session_state.emotion_history)}")
        st.write(f"CSV examples: {len(examples_df)}")
        if st.button("Clear History"):
            st.session_state.emotion_history = []
            st.rerun()
        if st.session_state.emotion_history:
            st.subheader("Recent Sessions")
            for item in reversed(st.session_state.emotion_history[-3:]):
                st.write(f"{item['field']}: {item['emotion']} ({item['confidence']:.1%})")

    st.title("Emotion-Aware Learning Assistant")
    st.caption("Detect learning emotions from text and generate supportive study guidance.")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Tell us about your learning challenge")
        name = st.text_input("Name", value="Demo User")
        email = st.text_input("Email", value="demo@example.com")
        role = st.selectbox("Role", ["student", "educator", "admin"])
        field = st.selectbox("What field are you studying?", ACADEMIC_FIELDS)
        problem = st.text_area(
            f"Describe your {field} problem or challenge:",
            placeholder=f"Example: I am confused about a topic in {field}.",
            height=140,
        )
        quick = st.columns(3)
        examples = [
            "I am confused about recursion",
            "Debugging is frustrating",
            "I am curious about machine learning",
        ]
        for column, text in zip(quick, examples):
            if column.button(text, use_container_width=True):
                problem = text

    with right:
        st.subheader("Settings")
        use_ai = st.checkbox("Use AI response with Gemini", value=True)
        save_data = st.checkbox("Save to CSV and database", value=True)
        show_details = st.checkbox("Show analysis details", value=False)
        use_csv_prediction = st.checkbox("Use CSV-based prediction hint", value=False)
        if use_csv_prediction and len(examples_df) > 0:
            st.info(f"Using {len(examples_df)} saved examples as reference data.")

    if st.button("Get AI Learning Help", type="primary", use_container_width=True):
        if not problem.strip():
            st.warning("Please describe the learning challenge first.")
            return

        upsert_user(email=email.strip(), name=name.strip() or "Learner", role=role)
        with st.spinner("Analyzing your learning state..."):
            bilstm_result = bilstm_model.predict(problem)
            bert_result = bert_model.predict(problem)
            selected = bilstm_result
            ai_response, response_type = (
                get_ai_response(field, problem, selected["emotion"], selected["confidence"])
                if use_ai
                else (EMOTION_RESPONSES[selected["emotion"]]["response"], "Template")
            )

        csv_logged = False
        if save_data:
            csv_logged = save_to_csv(field, problem, selected["emotion"], selected["confidence"], ai_response)

        for result in (bilstm_result, bert_result):
            add_history(field, problem, result, ai_response, response_type)
            if save_data:
                mixed = get_mixed_emotions(result["scores"], MIXED_EMOTION_THRESHOLD)
                insert_emotion_record(
                    email=email.strip(),
                    field=field,
                    input_text=problem,
                    predicted_emotion=result["emotion"],
                    secondary_emotion=mixed[1][0] if len(mixed) > 1 else None,
                    confidence_score=result["confidence"],
                    model_used=result["model"],
                    ai_response=ai_response,
                    response_type=response_type,
                    emotion_scores=result["scores"],
                    csv_logged=csv_logged,
                )

        st.subheader("Model Predictions Comparison")
        col1, col2 = st.columns(2)
        with col1:
            show_prediction("BiLSTM Student Adaptive", bilstm_result)
        with col2:
            show_prediction("BERT Transformer", bert_result)

        st.subheader("AI Learning Assistant Response")
        st.info(f"Response based on BiLSTM prediction: {selected['emotion']}")
        st.write(ai_response)
        st.subheader("Additional Support")
        st.success(f"Strategy: {EMOTION_RESPONSES[selected['emotion']]['action']}")

        if show_details:
            with st.expander("Analysis Details", expanded=True):
                st.write(f"Original problem: {problem}")
                st.write(f"BiLSTM processed: {bilstm_result['cleaned_text']}")
                st.write(f"BiLSTM confidence: {bilstm_result['confidence']:.3f}")
                st.write(f"Response type: {response_type}")
                st.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")

    show_analytics()


if __name__ == "__main__":
    main()
