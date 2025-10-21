# Imports
from __future__ import annotations

import pandas as pd
import streamlit as st

# Ensures VADER is available (downloads if missing)
def _ensure_vader():
    """Download the VADER lexicon if missing and create SIA."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        # Will raise LookupError if not present
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    return SentimentIntensityAnalyzer()

# Helper function to get VADER
@st.cache_resource(show_spinner=False)
def _get_vader():
    return _ensure_vader()

# Maps VADER's compoind score (-1 to 1) into labels: Positive, Neutral, or Negative
def _label_from_compound(c: float) -> str:
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

# Scores a list of news headlines using VADER sentiment analysis
def score_headlines_vader(texts: list[str]) -> pd.DataFrame:
    """
    Score a list of headlines with VADER.
    Returns DataFrame with columns: text, compound, label
    """
    if not texts:
        return pd.DataFrame(columns=["text", "compound", "label"])

    sia = _get_vader()
    rows = []
    for t in texts:
        t = t or ""
        comp = float(sia.polarity_scores(t)["compound"])
        rows.append({"text": t, "compound": comp, "label": _label_from_compound(comp)})

    return pd.DataFrame(rows, columns=["text", "compound", "label"])

# Aggregate sentiment results into overall counts and a net sentiment score
def summarize_labels(scored: pd.DataFrame) -> dict:
    """
    Summarize sentiment into counts and a net score in [-1, 1].
    - If 'compound' column exists, use its mean.
    - Otherwise compute score = (pos - neg) / total.
    """
    if scored is None or scored.empty:
        return {"positive": 0, "neutral": 0, "negative": 0, "majority": "neutral", "score": 0.0}

    counts = scored["label"].value_counts().to_dict()
    pos = int(counts.get("positive", 0))
    neu = int(counts.get("neutral", 0))
    neg = int(counts.get("negative", 0))
    total = max(1, pos + neu + neg)

    if "compound" in scored.columns:
        net = float(scored["compound"].mean())  # already in [-1, 1]
    else:
        net = (pos - neg) / total  # heuristic fallback

    majority = (
        "positive" if pos > max(neu, neg)
        else "negative" if neg > max(pos, neu)
        else "neutral"
    )

    return {"positive": pos, "neutral": neu, "negative": neg, "majority": majority, "score": net}


# If transformers/torch aren’t installed, this still imports fine; the app checks availability
_HAS_FINBERT_DEPS = True
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    _HAS_FINBERT_DEPS = False

# Loads and caches FinBERT 
@st.cache_resource(show_spinner=False)
def _load_finbert():
    """
    Cached load of ProsusAI/finbert.
    Returns (tokenizer, model) or raises if dependencies missing.
    """
    if not _HAS_FINBERT_DEPS:
        raise RuntimeError("FinBERT dependencies not available (install transformers + torch).")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Uses FinBERT to analyze headlines and computes probabilities for positive, neutral, and negative cases while computing a compound score
def score_headlines_finbert(texts: list[str]) -> pd.DataFrame:
    """
    Score a list of headlines with FinBERT.
    Returns DataFrame with columns: text, compound, label
    - label in {'negative','neutral','positive'}
    - compound ~= P(positive) - P(negative) in [-1, 1]
    """
    if not texts:
        return pd.DataFrame(columns=["text", "compound", "label"])

    tokenizer, model = _load_finbert()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  

    idx = probs.argmax(axis=1)                 
    labels = ["negative", "neutral", "positive"]
    mapped = [labels[i] for i in idx]
    compound = probs[:, 2] - probs[:, 0]       

    return pd.DataFrame({"text": texts, "compound": compound, "label": mapped})
