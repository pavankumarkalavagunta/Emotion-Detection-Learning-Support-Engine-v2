from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from src.bert_model import BERTEmotionClassifier
from src.config import ACADEMIC_FIELDS, EMOTIONS, EXAMPLES_CSV, MAPPING_CSV, MIXED_EMOTION_THRESHOLD, ensure_project_dirs
from src.database import insert_emotion_record, upsert_user
from src.preprocessing import EMOTION_RESPONSES, EmotionPredictor, get_mixed_emotions

ensure_project_dirs()


st.set_page_config(
    page_title="Emotion Detection Learning Support Engine v2",
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


def get_ai_response(field: str, problem: str, emotion: str, confidence: float) -> tuple[str, str]:
    return EMOTION_RESPONSES[emotion]["response"], "Template"


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
        "prediction_time": result.get("prediction_time", 0.0),
    }
    st.session_state.emotion_history.append(entry)


def show_prediction(title: str, result: dict) -> None:
    st.write(f"**{title}**")
    mixed = get_mixed_emotions(result["scores"], MIXED_EMOTION_THRESHOLD)
    if len(mixed) > 1:
        st.metric("Mixed Emotions", " + ".join(item[0] for item in mixed), f"Primary: {mixed[0][1]:.1%}")
    else:
        st.metric("Emotion", result["emotion"], f"{result['confidence']:.1%}")
    st.metric("Prediction Time", f"{result.get('prediction_time', 0.0):.3f}s")

    chart_df = pd.DataFrame(
        [{"Emotion": emotion, "Probability": float(result["scores"].get(emotion, 0.0))} for emotion in EMOTIONS]
    ).set_index("Emotion")
    st.bar_chart(chart_df, y="Probability")
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
            st.bar_chart(counts)
        with col2:
            st.line_chart(df.set_index("time")["confidence"])

    with tab2:
        grouped = df.groupby(["field", "emotion", "model"]).size().reset_index(name="count")
        st.dataframe(grouped, use_container_width=True)

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

    st.title("Emotion Detection Learning Support Engine v2")
    st.caption("Detect Happy, Sad, Angry, Fear, Surprise, Love, and Neutral emotions from learning text.")

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
        model_choice = st.radio("Model", ["Use BERT", "Use BiLSTM", "Compare Both Models"], index=0)
        save_data = st.checkbox("Save to CSV and database", value=True)
        show_details = st.checkbox("Show analysis details", value=False)

    if st.button("Get AI Learning Help", type="primary", use_container_width=True):
        if not problem.strip():
            st.warning("Please describe the learning challenge first.")
            return

        if save_data:
            upsert_user(email=email.strip(), name=name.strip() or "Learner", role=role)
        with st.spinner("Analyzing your learning state..."):
            bert_start = time.perf_counter()
            bert_result = bert_model.predict(problem)
            bert_result["prediction_time"] = time.perf_counter() - bert_start
            bilstm_start = time.perf_counter()
            bilstm_result = bilstm_model.predict(problem)
            bilstm_result["prediction_time"] = time.perf_counter() - bilstm_start
            if model_choice == "Use BERT":
                visible_results = [bert_result]
                selected = bert_result
            elif model_choice == "Use BiLSTM":
                visible_results = [bilstm_result]
                selected = bilstm_result
            else:
                visible_results = [bert_result, bilstm_result]
                selected = max(visible_results, key=lambda item: item["confidence"])
            ai_response, response_type = get_ai_response(field, problem, selected["emotion"], selected["confidence"])

        csv_logged = False
        if save_data:
            csv_logged = save_to_csv(field, problem, selected["emotion"], selected["confidence"], ai_response)

        for result in visible_results:
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

        st.subheader("Model Prediction")
        columns = st.columns(len(visible_results))
        for column, result in zip(columns, visible_results):
            with column:
                show_prediction(result["model"], result)

        st.subheader("AI Learning Assistant Response")
        st.info(f"Response based on {selected['model']} prediction: {selected['emotion']}")
        st.write(ai_response)
        st.subheader("Input Text")
        st.write(problem)
        st.subheader("Additional Support")
        st.success(f"Strategy: {EMOTION_RESPONSES[selected['emotion']]['action']}")

        if show_details:
            with st.expander("Analysis Details", expanded=True):
                st.write(f"Original problem: {problem}")
                st.write(f"Processed text: {selected['cleaned_text']}")
                st.write(f"Confidence: {selected['confidence']:.3f}")
                st.write(f"Prediction time: {selected['prediction_time']:.3f}s")
                st.write(f"Response type: {response_type}")
                st.write(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")

    show_analytics()


if __name__ == "__main__":
    main()
