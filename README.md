# Emotion-Aware Learning Assistant

A production-ready **Streamlit** application that detects learning-related emotions from text using trained **BiLSTM** and **BERT** models, and provides AI-powered supportive study guidance.

## ✨ Features

- **Dual Model Emotion Detection**
  - BiLSTM: Efficient LSTM-based neural network for real-time emotion classification
  - BERT: State-of-the-art transformer model fine-tuned for emotion detection
  - Side-by-side model comparison and prediction confidence

- **7-Class Emotion Recognition**
  - Sadness, Joy, Love, Anger, Fear, Surprise, Neutral

- **AI-Powered Study Guidance**
  - Gemini API integration for personalized learning assistance (fallback to templates)
  - Context-aware responses based on detected emotions and study field
  - Supportive coaching strategies for each emotion

- **Learning Analytics Dashboard**
  - Emotion distribution analysis
  - Study field breakdown
  - Model performance comparison
  - Confidence trends over time

- **Data Persistence**
  - SQLite database for detailed records
  - CSV logging for analytics
  - User profiles with role-based support

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- 2GB+ RAM (4GB+ recommended for model training)
- GPU (NVIDIA CUDA) optional but recommended for faster training

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd Emotion Detection & Learning Support Engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Prepare Data

The models will automatically download the `dair-ai/emotion` dataset from Hugging Face during training. No manual data preparation needed!

### 3. Train Models

**Important:** Both models must be trained before running the Streamlit app.

#### Train BiLSTM Model
```bash
python train_bilstm.py
```

This will:
- Download the dair-ai/emotion dataset (first run only)
- Preprocess and clean all texts
- Train the BiLSTM model for 3 epochs
- Save model, tokenizer, and label encoder to `models/bilstm/`
- Display training and validation metrics

Expected output:
```
============================================================
BiLSTM Emotion Detection Model Training
============================================================
Loading dair-ai/emotion dataset...
Training samples: 16000
Validation samples: 2000
Test samples: 2000

Building BiLSTM model...
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
...

Training for 3 epochs...
Epoch 1/3
500/500 [...] loss: 0.5234 - accuracy: 0.8123 - val_loss: 0.4876 - val_accuracy: 0.8456

============================================================
✓ BiLSTM Model Training Complete!
============================================================
```

#### Train BERT Model
```bash
python train_bert.py
```

This will:
- Download bert-base-uncased from Hugging Face
- Fine-tune on the emotion dataset for 3 epochs
- Save fine-tuned model to `models/bert_emotion_model_final/`
- Display training and evaluation metrics

Expected output:
```
============================================================
BERT Fine-tuning for Emotion Detection
============================================================
Loading dair-ai/emotion dataset...
Dataset splits: dict_keys(['train', 'validation', 'test'])

Loading bert-base-uncased...

Tokenizing dataset...
Tokenized training set size: 16000
Tokenized validation set size: 2000
Tokenized test set size: 2000

Starting fine-tuning training...

[=============================>] Epoch 1/3

Test Results:
  accuracy: 0.8876
  f1: 0.8821
  precision: 0.8945
  recall: 0.8876

============================================================
✓ BERT Fine-tuning Complete!
Model saved to: models/bert_emotion_model_final
============================================================
```

### 4. Evaluate Models (Optional)

```bash
python evaluate_models.py
```

This generates:
- Detailed accuracy, precision, recall, F1-scores
- Classification reports for both models
- Confusion matrices
- Performance comparison
- Results saved to `evaluation_results.json`

### 5. Run Streamlit Application

```bash
streamlit run app.py
```

The app will:
- Load both trained models (errors if not trained)
- Open in your browser at `http://localhost:8501`
- Display "✓ BiLSTM Model Loaded" and "✓ BERT Model Loaded" in the sidebar

## 📁 Project Structure

```
project/
├── app.py                          # Main Streamlit application
├── train_bilstm.py                # BiLSTM training script
├── train_bert.py                  # BERT fine-tuning script
├── evaluate_models.py             # Model evaluation script
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration and constants
│   ├── preprocessing.py           # Text preprocessing pipeline
│   ├── bilstm_model.py           # BiLSTM model and predictor
│   ├── bert_model.py             # BERT model and predictor
│   ├── database.py               # SQLite database operations
│
├── models/
│   ├── bilstm/
│   │   ├── bilstm_model.keras    # Trained BiLSTM model
│   │   ├── tokenizer.pkl         # Word tokenizer
│   │   ├── label_encoder.pkl     # Label encoder
│   │   └── emotions.json         # Emotion classes
│   │
│   └── bert_emotion_model_final/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── special_tokens_map.json
│       ├── model.safetensors
│       └── label_mapping.json
│
├── data/
│   └── learning_assistant.db      # SQLite database (auto-created)
│
├── notebooks/
│   └── training_pipeline.ipynb    # Detailed training walkthrough
│
├── requirements.txt               # Python dependencies
├── .streamlit/config.toml        # Streamlit configuration
├── .env.example                   # Environment template
├── DEPLOYMENT_GUIDE.md           # Cloud deployment instructions
├── README.md                      # This file
└── schema.sql                     # Database schema
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: Gemini API key for AI study guidance
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Google API key (alternative to GEMINI_API_KEY)
GOOGLE_API_KEY=your_google_api_key_here
```

Get your Gemini API key from: https://aistudio.google.com/app/apikeys

### Model Hyperparameters

Edit `src/config.py` to adjust:

```python
# BiLSTM
VOCAB_SIZE = 10000        # Max vocabulary size
EMBEDDING_DIM = 100       # Embedding dimension
LSTM_UNITS = 128          # LSTM hidden units
BATCH_SIZE = 32           # Training batch size
EPOCHS = 3                # Number of training epochs

# BERT
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LENGTH = 128     # Max sequence length

# General
MAX_SEQ_LEN = 128
MIXED_EMOTION_THRESHOLD = 0.15  # Threshold for multi-emotion detection
```

## 📊 Database Schema

The SQLite database stores:

**Users Table:**
- email (primary key)
- name, role
- created_at, updated_at

**Emotion Records Table:**
- record_id (primary key)
- email (foreign key)
- field, input_text
- predicted_emotion, secondary_emotion, confidence_score
- model_used, ai_response, response_type
- emotion_scores (JSON), csv_logged
- timestamp

## 🎓 Emotion Classes

| Emotion | Label | Description |
|---------|-------|-------------|
| sadness | 😢 Sad | Feeling down, discouraged |
| joy | 😊 Happy | Excited, pleased, satisfied |
| love | ❤️ Love | Passionate, caring, enthusiastic |
| anger | 😠 Angry | Frustrated, upset, irritated |
| fear | 😨 Fear | Anxious, worried, uncertain |
| surprise | 😲 Surprised | Amazed, astonished, caught off-guard |
| neutral | 😐 Neutral | Calm, composed, unaffected |

## 🚀 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Set environment variables in the cloud dashboard
5. Deploy!

See `DEPLOYMENT_GUIDE.md` for Google Cloud Run, AWS, and Azure instructions.

## 📈 Model Performance

### BiLSTM
- **Architecture**: Embedding → Bidirectional LSTM → Bidirectional LSTM → Dense layers → Softmax
- **Training time**: ~5-10 minutes on CPU, ~2-3 minutes on GPU
- **Parameters**: ~1.2M

### BERT
- **Model**: bert-base-uncased fine-tuned for emotion classification
- **Training time**: ~15-30 minutes on CPU, ~5-10 minutes on GPU
- **Parameters**: ~110M

## 🛠️ Development

### Running Tests

```bash
# Evaluate both models on test set
python evaluate_models.py

# Test individual model predictions
python -c "from src.bilstm_model import BiLSTMPredictor; p = BiLSTMPredictor(); print(p.predict('I am confused'))"
python -c "from src.bert_model import BERTEmotionClassifier; b = BERTEmotionClassifier(); print(b.predict('I am confused'))"
```

### Code Quality

```bash
# Format code
black . --line-length 100

# Lint
flake8 . --max-line-length 100 --ignore E203,W503

# Type checking (optional)
mypy src/
```

## 📝 API Reference

### BiLSTMPredictor

```python
from src.bilstm_model import BiLSTMPredictor

predictor = BiLSTMPredictor()
result = predictor.predict("I'm confused about recursion")

# Returns:
# {
#     "emotion": "fear",
#     "confidence": 0.87,
#     "scores": {
#         "sadness": 0.05,
#         "joy": 0.01,
#         ...
#         "fear": 0.87
#     },
#     "model": "BiLSTM"
# }
```

### BERTEmotionClassifier

```python
from src.bert_model import BERTEmotionClassifier

classifier = BERTEmotionClassifier()
result = classifier.predict("I'm confused about recursion")

# Same return format as BiLSTMPredictor
```

### TextPreprocessor

```python
from src.preprocessing import TextPreprocessor

# Clean text
cleaned = TextPreprocessor.clean_text("Check out https://example.com 😊!")
# Returns: "check out"

# Tokenize and pad
padded, tokenizer = TextPreprocessor.tokenize_and_pad(
    texts=["sample text"],
    fit_on_texts=True
)
```

## 🐛 Troubleshooting

### Models not found error
```
FileNotFoundError: Model or tokenizer not found
```
**Solution**: Run `python train_bilstm.py` and `python train_bert.py`

### CUDA out of memory
**Solution**: Reduce `BATCH_SIZE` in `src/config.py` or train on CPU

### Streamlit import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Database locked error
**Solution**: Delete `learning_assistant.db-journal` and restart

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [dair-ai/emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

- Check `DEPLOYMENT_GUIDE.md` for deployment questions
- Review `PROJECT_ANALYSIS_REPORT.md` for technical details
- Open an issue on GitHub for bugs

---

**Built with** ❤️ using Streamlit, TensorFlow, PyTorch, and Hugging Face Transformers
