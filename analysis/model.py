# Imports
import numpy as np
from core.features import momentum_features

# Calculates an overall technical analysis score using weighted components from moving averages, momentum, and slope
def compute_technical_score(df):
    f = momentum_features(df)
    cross_score = 40 * f["cross"]
    momentum_score = 8 * f["momentum"]
    slope_score = 10 * f["slope"]
    total = np.clip(cross_score + momentum_score + slope_score, -100, 100)
    return float(total), f

# Coverts a normalized sentiment score (-1 to +1) into the same -100 to +100 range for fusion
def compute_sentiment_score(summary):
    return float(np.clip(summary.get("score", 0.0) * 100, -100, 100))

# Blends technical and sentiment signals into unified trading indicator. 
def fuse_signal(tech, senti, w_tech=0.6, w_senti=0.4):
    fused = np.clip(w_tech * tech + w_senti * senti, -100, 100)
    if fused >= 20:
        label = "BUY"
    elif fused <= -20:
        label = "SELL"
    else:
        label = "HOLD"
    confidence = (
        "High" if abs(fused) >= 60 else
        "Medium" if abs(fused) >= 35 else
        "Low"
    )
    return label, fused, confidence
