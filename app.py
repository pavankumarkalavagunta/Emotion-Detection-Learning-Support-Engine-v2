from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.bert_model import BERTEmotionClassifier
from src.bilstm_model import BiLSTMPredictor
from src.config import (
    ACADEMIC_FIELDS,
    BERT_DIR,
    BILSTM_DIR,
    EMOTIONS,
    EMOTION_LABELS,
    EXAMPLES_CSV,
    MAPPING_CSV,
    MIXED_EMOTION_THRESHOLD,
)
from src.database import insert_emotion_record, upsert_user
from src.preprocessing import EMOTION_RESPONSES, get_mixed_emotions

load_dotenv()


st.set_page_config(
    page_title="Emotion-Aware Learning Assistant",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_models() -> tuple[BiLSTMPredictor, BERTEmotionClassifier]:
    """Load both trained models. Raises error if not trained."""
    try:
        bilstm = BiLSTMPredictor(BILSTM_DIR)
        st.sidebar.success("✓ BiLSTM Model Loaded")
    except FileNotFoundError as e:
        st.sidebar.error(f"✗ BiLSTM Model Error: {str(e)}")
        st.error("BiLSTM model not found. Please run `python train_bilstm.py` first.")
        st.stop()
    except RuntimeError as e:
        st.sidebar.error(f"✗ BiLSTM Error: {str(e)}")
        st.error(f"Failed to load BiLSTM model: {str(e)}")
        st.stop()

    try:
        bert = BERTEmotionClassifier(BERT_DIR)
        st.sidebar.success("✓ BERT Model Loaded")
    except FileNotFoundError as e:
        st.sidebar.error(f"✗ BERT Model Error: {str(e)}")
        st.error("BERT model not found. Please run `python train_bert.py` first.")
        st.stop()
    except RuntimeError as e:
        st.sidebar.error(f"✗ BERT Error: {str(e)}")
        st.error(f"Failed to load BERT model: {str(e)}")
        st.stop()

    return bilstm, bert


@st.cache_data
def load_examples() -> pd.DataFrame:
    """Load saved emotion examples."""
    if EXAMPLES_CSV.exists():
        return pd.read_csv(EXAMPLES_CSV)
    return pd.DataFrame(
        columns=["text", "emotion", "confidence", "response", "field", "timestamp"]
    )


def configure_gemini():
    """Configure Gemini API client for study advice generation."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        return genai.Client(api_key=api_key)
    except Exception:
        return None


def get_ai_response(
    field: str, problem: str, emotion: str, confidence: float
) -> tuple[str, str]:
    """
    Generate AI-powered study advice using Gemini.
    Falls back to template if API unavailable.
    """
    model = configure_gemini()
    fallback = EMOTION_RESPONSES[emotion]["response"]

    if model is None:
        return fallback, "Template"

    prompt = f"""You are a supportive, empathetic learning assistant helping a student with {field}.

The student is currently feeling {emotion} (detected with {confidence:.1%} confidence).
Their challenge: {problem}

Provide a brief, encouraging response that:
1. Acknowledges their emotional state
2. Offers 1-2 specific strategies for {field}
3. Suggests their next learning step

Keep the response concise, supportive, and focused on helping them progress."""

    try:
        response = model.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text.strip(), "Gemini AI"
    except Exception:
        return fallback, "Template"


def save_to_csv(
    field: str, problem: str, emotion: str, confidence: float, response: str
) -> bool:
    """Save emotion analysis to CSV files."""
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
                mapping = pd.concat(
                    [mapping, pd.DataFrame([mapping_row])], ignore_index=True
                )
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
    """Add prediction to session history."""
    mixed = get_mixed_emotions(result["scores"], MIXED_EMOTION_THRESHOLD)
    emotion_label = (
        " + ".join(item[0] for item in mixed) if len(mixed) > 1 else result["emotion"]
    )
    secondary = mixed[1][0] if len(mixed) > 1 else None
    entry = {
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion_label": emotion_label,
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
    """Display emotion prediction results."""
    st.write(f"### {title}")
    mixed = get_mixed_emotions(result["scores"], MIXED_EMOTION_THRESHOLD)

    if len(mixed) > 1:
        emotion_display = " + ".join(item[0] for item in mixed)
        confidence_display = f"{mixed[0][1]:.1%} (primary)"
    else:
        emotion_display = EMOTION_LABELS.get(result["emotion"], result["emotion"])
        confidence_display = f"{result['confidence']:.1%}"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Detected Emotion:** {emotion_display}")
    with col2:
        st.metric("Confidence", confidence_display)

    # Show probability distribution
    st.write("**Probability Distribution:**")
    for emotion, score in sorted(
        result["scores"].items(), key=lambda item: item[1], reverse=True
    ):
        st.progress(float(score), text=f"{EMOTION_LABELS.get(emotion, emotion)}: {score:.1%}")




def show_analytics() -> None:
    """Display learning analytics dashboard."""
    history = st.session_state.emotion_history
    if not history:
        return

    st.divider()
    st.header("📊 Learning Analytics Dashboard")
    df = pd.DataFrame(history)
    df["time"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M:%S")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Emotions 📈", "Fields 📚", "Model Comparison 🤖", "Summary 📋"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            counts = df["emotion_label"].value_counts()
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                title="Emotion Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(
                df,
                x="time",
                y="confidence",
                color="emotion_label",
                markers=True,
                title="Confidence Over Time",
                line_shape="spline",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        grouped = df.groupby(["field", "emotion_label"]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x="field",
            y="count",
            color="emotion_label",
            title="Emotions by Study Field",
            barmode="stack",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        model_comparison = df.groupby("model")["confidence"].agg(
            ["mean", "count", "min", "max"]
        )
        st.dataframe(model_comparison, use_container_width=True)

        fig = px.box(
            df, x="model", y="confidence", title="Confidence Distribution by Model"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Interactions", len(df))
        col2.metric("Average Confidence", f"{df['confidence'].mean():.1%}")
        col3.metric("Most Common Emotion", df["emotion_label"].mode().iloc[0])
        col4.metric("Primary Models Used", df["model"].nunique())

        st.write("**Recent Interactions:**")
        st.dataframe(
            df[
                [
                    "timestamp",
                    "field",
                    "emotion_label",
                    "confidence",
                    "model",
                    "response_type",
                ]
            ].tail(10),
            use_container_width=True,
        )


def main() -> None:
    """Main application logic."""
    # Initialize session state
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []

    # Load models (will stop if they don't exist)
    bilstm_model, bert_model = load_models()
    examples_df = load_examples()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard")
        st.write("Models: ✓ BiLSTM + BERT")
        st.write(f"Interactions: {len(st.session_state.emotion_history)}")
        st.write(f"Saved Examples: {len(examples_df)}")

        if st.button("🗑️ Clear History"):
            st.session_state.emotion_history = []
            st.rerun()

        if st.session_state.emotion_history:
            st.subheader("Recent Sessions")
            for item in reversed(st.session_state.emotion_history[-3:]):
                st.write(
                    f"**{item['field']}:** {item['emotion_label']} ({item['confidence']:.1%})"
                )

    # Main content
    st.title("🧠 Emotion-Aware Learning Assistant")
    st.caption(
        "Detect your learning emotions using AI and receive personalized study guidance."
    )

    # User input section
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Share Your Learning Challenge")
        name = st.text_input("Your Name", value="Learner", key="name_input")
        email = st.text_input("Email", value="learner@example.com", key="email_input")
        role = st.selectbox("Role", ["student", "educator", "parent", "other"])
        field = st.selectbox("Study Field", ACADEMIC_FIELDS)

        problem = st.text_area(
            f"Describe your {field} challenge:",
            placeholder=f"Example: I'm struggling to understand the concept of recursion in {field}.",
            height=140,
            key="problem_input",
        )

        # Quick example buttons
        st.write("**Quick Examples:**")
        quick_cols = st.columns(3)
        examples = [
            "I don't understand this concept at all",
            "I'm frustrated with debugging",
            "I'm curious about how this works",
        ]
        for col, example_text in zip(quick_cols, examples):
            if col.button(example_text, use_container_width=True, key=f"btn_{example_text}"):
                problem = example_text

    with right:
        st.subheader("⚙️ Settings")
        use_ai = st.checkbox("Use Gemini AI Advice", value=True)
        save_data = st.checkbox("Save Results", value=True)
        show_details = st.checkbox("Show Details", value=False)

    # Prediction button
    if st.button("🎯 Analyze My Emotions", type="primary", use_container_width=True):
        if not problem.strip():
            st.warning("⚠️ Please describe your learning challenge first.")
            st.stop()

        # Update user
        upsert_user(
            email=email.strip(), name=name.strip() or "Learner", role=role
        )

        # Get predictions from both models
        with st.spinner("🔍 Analyzing your emotions..."):
            try:
                bilstm_result = bilstm_model.predict(problem)
                bert_result = bert_model.predict(problem)
            except RuntimeError as e:
                st.error(f"Prediction failed: {str(e)}")
                st.stop()

        # Save to database and CSV
        csv_logged = False
        if save_data:
            csv_logged = save_to_csv(
                field, problem, bilstm_result["emotion"],
                bilstm_result["confidence"],
                ""  # Response will be filled later
            )

        # Generate AI response
        if use_ai:
            ai_response, response_type = get_ai_response(
                field, problem, bilstm_result["emotion"], bilstm_result["confidence"]
            )
        else:
            ai_response = EMOTION_RESPONSES[bilstm_result["emotion"]]["response"]
            response_type = "Template"

        # Add to history
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

        # Display results
        st.divider()
        st.subheader("🤖 Model Predictions Comparison")
        col1, col2 = st.columns(2)
        with col1:
            show_prediction("BiLSTM Model", bilstm_result)
        with col2:
            show_prediction("BERT Model", bert_result)

        # AI Response
        st.divider()
        st.subheader("💡 Your Personalized Guidance")
        primary_emotion = EMOTION_LABELS.get(
            bilstm_result["emotion"], bilstm_result["emotion"]
        )
        st.info(f"Based on detected emotion: **{primary_emotion}**")
        st.write(ai_response)

        st.success(
            f"**Next Step:** {EMOTION_RESPONSES[bilstm_result['emotion']]['action']}"
        )

        # Detailed analysis
        if show_details:
            with st.expander("📋 Detailed Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**BiLSTM Details:**")
                    st.json(bilstm_result)
                with col2:
                    st.write("**BERT Details:**")
                    st.json(bert_result)

    # Show analytics
    show_analytics()

    # Footer
    st.divider()
    st.caption(
        "Powered by BiLSTM and BERT • Built with Streamlit • "
        "📧 Questions? Check the README for support."
    )


if __name__ == "__main__":
    main()

