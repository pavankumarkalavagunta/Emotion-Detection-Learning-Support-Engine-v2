# Emotion Detection Learning Support Engine v2

A production-ready Streamlit AI project for detecting learner emotions from text and returning supportive learning recommendations. The project supports training from scratch with both HuggingFace BERT and TensorFlow/Keras BiLSTM models.

## Supported Emotions

Happy, Sad, Angry, Fear, Surprise, Love, Neutral

## Folder Structure

```text
app.py
runtime.txt
requirements.txt
dataset/
models/
  bert_emotion_model_final/
  bilstm/
training/
  preprocess.py
  train_bilstm.py
  train_bert.py
  evaluate.py
  inference.py
utils/
checkpoints/
reports/
src/
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Format

Place a CSV file at `dataset/emotion_dataset.csv` with these columns:

```csv
text,emotion
"I am excited to solve this problem",Happy
"I am nervous about the exam",Fear
```

Labels must be one of: `Happy`, `Sad`, `Angry`, `Fear`, `Surprise`, `Love`, `Neutral`.

## Preprocessing

The training pipeline lowercases text, removes URLs, punctuation and emojis, expands contractions, removes stopwords, lemmatizes, tokenizes, and creates train/validation splits.

```powershell
python -m training.preprocess --csv dataset/emotion_dataset.csv
```

## Training

Train the BiLSTM model:

```powershell
python -m training.train_bilstm --csv dataset/emotion_dataset.csv --epochs 10 --batch-size 32
```

Outputs:

```text
models/bilstm/bilstm_model.keras
models/bilstm/tokenizer.json
models/bilstm/label_classes.json
```

Train the BERT model:

```powershell
python -m training.train_bert --csv dataset/emotion_dataset.csv --epochs 3 --batch-size 8
```

Outputs:

```text
models/bert_emotion_model_final/
```

## Evaluation Artifacts

Training writes evaluation files into `reports/`:

```text
classification_report.txt
confusion_matrix.png
training_accuracy.png
training_loss.png
metrics.json
```

## Inference

```powershell
python -m training.inference "I am anxious but motivated to learn" --model compare
```

## Streamlit App

`app.py` is the Streamlit Cloud entry point.

```powershell
streamlit run app.py
```

The app lets users select `Use BERT`, `Use BiLSTM`, or `Compare Both Models`, then displays predicted emotion, confidence, probability chart, prediction time, input text, and a learning recommendation.

## Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Ensure `runtime.txt` contains `python-3.11`.
3. Ensure `requirements.txt` is committed.
4. In Streamlit Cloud, choose this repository and set the main file path to `app.py`.
5. Commit trained model artifacts under `models/` or attach them using your preferred model artifact workflow.

The app starts even if trained model files are absent; it uses a deterministic fallback so deployment can be validated before model training finishes.

## Model Architecture

BERT uses `BertTokenizer` and `BertForSequenceClassification` with validation, early stopping, scheduler support through `Trainer`, and weighted accuracy, precision, recall, and F1 metrics.

BiLSTM uses Embedding, Bidirectional LSTM, Dropout, Dense, and Softmax layers. It trains with categorical cross entropy and saves a `.keras` model plus tokenizer metadata.

## Screenshots

Add screenshots after deployment, for example:

```text
docs/screenshots/streamlit-home.png
docs/screenshots/prediction-results.png
```
