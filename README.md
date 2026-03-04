# AI Powered Unified Customer Complaint Communication Dashboard for Banking

## Deployable Hackathon Architecture

| Feature | Tool |
|---|---|
| Complaint classification | HuggingFace small model (optional) + rule fallback |
| Sentiment analysis | TextBlob |
| Duplicate detection | Sentence Transformers (optional) / TF-IDF fallback |
| Response generation | Template + small local LLM (optional) |
| Dashboard | Streamlit |

## Run

```bash
pip install -r requirements.txt
python app.py --reset
python -m streamlit run dashboard/dashboard_ui.py
```

## Optional lightweight AI toggles

Enable these only when you want local model inference:

```bash
# HuggingFace zero-shot classifier
set USE_HF_CLASSIFIER=1
set HF_CLASSIFIER_MODEL=typeform/distilbert-base-uncased-mnli

# Small local response LLM
set USE_SMALL_LLM=1
set LOCAL_LLM_MODEL=google/flan-t5-small

# Sentence-transformer duplicates
set USE_SENTENCE_TRANSFORMERS=1
```

Default behavior is optimized for stable deployment and quick startup, with deterministic fallbacks.
