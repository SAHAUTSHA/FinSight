# Imports
from __future__ import annotations
import pandas as pd
import numpy as np

# Generates a compact set of technical indicators from a price DataFrame with a 'close' column
def build_features(
    prices: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50
) -> pd.DataFrame:
    """
    Given a price DataFrame with a 'close' column, compute a small set of
    interpretable technical features used by the Signal panel.

    Returns the original df with extra columns:
      - sma_fast, sma_slow
      - sma_cross (1 bullish cross, -1 bearish cross, 0 otherwise)
      - slope (normalized slope of slow SMA)
      - momentum (normalized recent percent change)
      - tech_score (0..+100)
    """
    df = prices.copy()

    if df.empty or "close" not in df.columns:
        return df.assign(
            sma_fast=np.nan,
            sma_slow=np.nan,
            sma_cross=0,
            slope=0.0,
            momentum=0.0,
            tech_score=50.0,
        )

    # SMAs
    df["sma_fast"] = df["close"].rolling(sma_fast, min_periods=1).mean()
    df["sma_slow"] = df["close"].rolling(sma_slow, min_periods=1).mean()

    # Cross signal: 1 when fast crosses above slow, -1 when crosses below
    cross = np.sign(df["sma_fast"] - df["sma_slow"])
    df["sma_cross"] = cross.diff().fillna(0).replace({np.inf: 0, -np.inf: 0}).clip(-1, 1)

    # Slope of slow SMA (normalized)
    df["slope"] = df["sma_slow"].diff().rolling(5, min_periods=1).mean()
    # Normalize slope to ~[-1,1] range to be comparable across tickers
    denom = (df["close"].rolling(sma_slow, min_periods=1).std() + 1e-9)
    df["slope"] = (df["slope"] / denom).clip(-1, 1).fillna(0.0)

    # Momentum (normalized recent % change over ~1 month if daily)
    df["momentum"] = (df["close"].pct_change(20)).clip(-0.3, 0.3).fillna(0.0) / 0.3
    df["momentum"] = df["momentum"].clip(-1, 1)

    # Simple, interpretable technical score in 0..100
    w_cross, w_slope, w_mom = 0.4, 0.3, 0.3
    tech_z = (
        w_cross * df["sma_cross"] +
        w_slope * df["slope"] +
        w_mom   * df["momentum"]
    ).clip(-1, 1)

    # Rescale to 0 to 100 for UI bars asthetics
    df["tech_score"] = ((tech_z + 1.0) / 2.0 * 100.0).round(1)

    return df

# Function that combines technical score and news sentiment score into a single interpretable market signal 
def fuse_signal(
    tech_score: float | int | None,
    news_score: float | int | None,
    tech_weight: float = 0.6,
) -> dict:
    """
    Blend a 0..100 technical score with a 0..100 news score to produce:
      - composite: 0..100
      - label: 'BUY' | 'HOLD' | 'SELL'
      - confidence: 'Low' | 'Medium' | 'High'
      - details: dict breakdown

    If one score is missing, it gracefully falls back to the other.
    """
    def _safe(x, default=None):
        return default if x is None or (isinstance(x, float) and np.isnan(x)) else x

    tech = _safe(float(tech_score), 50.0)
    news = _safe(float(news_score), 50.0)

    tech_w = float(tech_weight)
    news_w = 1.0 - tech_w
    composite = tech_w * tech + news_w * news

    # Discrete label bands
    if composite >= 66:
        label = "BUY"
    elif composite <= 34:
        label = "SELL"
    else:
        label = "HOLD"

    # Confidence from dispersion of components
    spread = abs(tech - news)
    if spread <= 8:
        confidence = "High"
    elif spread <= 20:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "composite": round(float(composite), 1),
        "label": label,
        "confidence": confidence,
        "details": {
            "tech_score": round(float(tech), 1),
            "news_score": round(float(news), 1),
            "weights": {"tech": tech_w, "news": news_w},
            "spread": round(float(spread), 1),
        },
    }
