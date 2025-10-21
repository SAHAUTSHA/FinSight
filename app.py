# Utsha Kumar Saha
# Project: FinSight 

# Imports
from __future__ import annotations
from typing import List, Dict, Tuple
import re
import pandas as pd
import streamlit as st
import yfinance as yf 

import nltk
nltk.download("vader_lexicon", quiet=True)


# Function Imports
from data.fetch_prices import get_prices_yf as get_prices
from data.fetch_news import get_news_newsapi
from viz.charts import line_chart, candlestick_chart
from viz.signal import render_signal_panel
from analysis.sentiment import score_headlines_vader, score_headlines_finbert

# Function to clean DataFrame columns (lowercase)
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df = df.rename(columns=lambda c: str(c).lower())
    return df

# Function that standardizes inconsistent API responses into uniform structure to display/analyze
def normalize_news(news_obj):
    if news_obj is None:
        return []
    if isinstance(news_obj, pd.DataFrame):
        rows = news_obj.to_dict(orient="records")
    elif isinstance(news_obj, (list, tuple)):
        rows = list(news_obj)
    else:
        rows = []

    out = []
    for r in rows:
        if isinstance(r, dict) and r.get("title"):
            out.append({
                "title": r.get("title"),
                "source": r.get("source") or r.get("source_name") or "",
                "url": r.get("url") or r.get("link") or "",
                "published_at": r.get("published_at") or r.get("published") or r.get("date"),
                "description": r.get("description") or r.get("summary") or "",
            })
    return out

# Function that formats sentiment label to an HTML label (sentiment results display ready)
def sentiment_badge(label: str) -> str:
    label_uc = (label or "").strip().capitalize()
    if label_uc == "Positive":
        bg, fg = "#0E4429", "#8CE99A"
    elif label_uc == "Negative":
        bg, fg = "#5A1A1A", "#FF9AA2"
    else:
        bg, fg = "#3A3A3A", "#EAEAEA"; label_uc = "Neutral"
    return f"""<span style="padding:4px 10px;border-radius:8px;background:{bg};color:{fg};
               font-weight:600;font-size:12px;display:inline-block;line-height:1.1;">{label_uc}</span>"""

# Function to count how many headlines are positive, neutral, and negative and calculates "net sentiment" score  
def summarize_label_list(labels: List[str]) -> Dict:
    labs = [(l or "").lower() for l in (labels or [])]
    pos = sum(1 for l in labs if l == "positive")
    neu = sum(1 for l in labs if l == "neutral")
    neg = sum(1 for l in labs if l == "negative")
    total = max(1, pos + neu + neg)
    net = (pos - neg) / total
    return {"positive": pos, "neutral": neu, "negative": neg, "net": float(net)}

# Decides if mood is Positive, Neutral, or Negative and computes if company is bullish or bearish 
# Results in HTML snippit displaying count, score, and color-coded badge
def news_sentiment_panel(labels: List[str]) -> Tuple[str, float]:
    s = summarize_label_list(labels)
    net = s["net"]
    overall = "Positive" if net > 0.05 else ("Negative" if net < -0.05 else "Neutral")
    counts_text = f"Counts: Positive {s['positive']} • Neutral {s['neutral']} • Negative {s['negative']}"
    delta = int(round(net * 100)); direction = "bullish" if delta >= 0 else "bearish"
    score_text = f"Net score: {delta:+d}% ({direction})"
    header_badge = sentiment_badge(overall)
    header = f"{header_badge}&nbsp;&nbsp;<span style='opacity:.9'>{counts_text}</span><br/><span style='opacity:.8'>{score_text}</span>"
    return header, (net + 1.0) / 2.0

# Function that converts Hex colors to RGB values
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# Function that converts RGB Values to Hex colors 
def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb

# Funtion that performs Linear Interpolation (finds a value between two numbers based on a percentage t)
def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

# Function that takes two colors and 'blends them' them together by a ratio 't'. 
def _blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1); r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((int(_lerp(r1, r2, t)), int(_lerp(g1, g2, t)), int(_lerp(b1, b2, t))))

# Function converts overall sentiment strength into precise color for charts and graphs (Red for bearish, Gray for neutral, and Green for bullish)
def sentiment_color_from_net(net: float) -> str:
    """
    net in [-1,1]; matches the BUY/SELL bubble colors.
    BUY green: #4CAF50, SELL red: #E53935, neutral gray: #9ca3af
    """
    red = "#E53935"
    gray = "#9ca3af"
    green = "#4CAF50"
    t = (net + 1.0) / 2.0
    if t < 0.5:
        return _blend(red, gray, t / 0.5)  # red → gray
    else:
        return _blend(gray, green, (t - 0.5) / 0.5)  # gray → green

# Function draws a horizontal progress bar in Streamlit using raw HTML. 
# Takes a percentage (from 0 to 1), a color, and an optional label and then renders a rounded bar with that percentage filled in the given color. 
def render_colored_bar(percent_0_1: float, color: str, label: str = ""):
    pct = max(0, min(100, int(round(percent_0_1 * 100))))
    st.markdown(
        f"""
        <div style="margin:12px 0 14px 0;">
          <div style="width:100%; height:14px; background:#e5e7eb; border-radius:999px;">
            <div style="width:{pct}%; height:100%; background:{color}; border-radius:999px;"></div>
          </div>
          <div style="font-size:.9rem; opacity:.8; margin-top:6px;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Function that cleans company names (removes legal suffixes)
def _strip_entity_suffixes(name: str) -> str:
    if not name:
        return ""
    suffixes = [
        r",?\s+inc\.?", r",?\s+incorporated", r",?\s+corporation", r",?\s+corp\.?",
        r",?\s+ltd\.?", r",?\s+limited", r",?\s+co\.?", r",?\s+plc", r",?\s+nv",
        r",?\s+sa", r",?\s+ag", r",?\s+holdings?", r",?\s+group"
    ]
    out = name.lower()
    for s in suffixes:
        out = re.sub(s + r"$", "", out, flags=re.I).strip()
    return out

# Function that finds the full company name for a given stock ticker (if not found uses Yahoo! Finance)
def resolve_company_name(ticker: str) -> str:
    """
    Resolve a user-friendly company name for the ticker.
    Uses yfinance as primary; falls back to a small alias map.
    """
    aliases = {
        "A": "Agilent Technologies, Inc.",
        "META": "Meta Platforms, Inc.",
        "GOOGL": "Alphabet Inc.",
        "BRK.B": "Berkshire Hathaway Inc.",
    }
    try:
        if ticker in aliases:
            return aliases[ticker]
        y = yf.Ticker(ticker)
        name = None
        try:
            info = y.get_info()
            name = info.get("longName") or info.get("shortName")
        except Exception:
            info = getattr(y, "info", {}) or {}
            name = info.get("longName") or info.get("shortName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return ticker

# Filters a list of news articles to find target company articles [Edge cases like 'A' or 'T']
def filter_news_for_company(items: List[Dict], company_name: str, ticker: str) -> List[Dict]:
    """
    Keep only articles that actually reference the company in title/description.
    We aggressively filter for single/short tickers.
    """
    if not items:
        return []

    stripped = _strip_entity_suffixes(company_name or "")
    variants = set()
    if company_name:
        variants.add(company_name.lower())
    if stripped and stripped != (company_name or "").lower():
        variants.add(stripped)

    cleaned_variants = set()
    for v in variants:
        cleaned_variants.add(re.sub(r"[,\.]", "", v))

    patterns = [re.compile(rf"\b{re.escape(v)}\b", flags=re.I) for v in cleaned_variants if v]
    ticker_ok = re.compile(rf"\b{re.escape(ticker)}\b", flags=re.I) if len(ticker) >= 3 else None

    filtered = []
    for it in items:
        text = (it.get("title") or "") + " " + (it.get("description") or "")
        text_low = re.sub(r"\s+", " ", text.lower()).strip()
        hit = any(p.search(text_low) for p in patterns)
        if not hit and ticker_ok:
            hit = bool(ticker_ok.search(text_low))
        if hit:
            filtered.append(it)

    if not filtered and len(ticker) >= 3:
        return items
    return filtered


# Page Configuration
st.set_page_config(page_title="FinSignal — Price meets sentiment", page_icon="📈", layout="wide")

# Page Style [CSS]
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"] {
      height: 100vh;
      overflow: hidden;
  }
  .block-container {
      height: 100%;
      padding-top: 50px !important;  /* space for top bar */
      padding-bottom: 0 !important;
  }
  .middle-fixed {
      position: fixed;
      left: 25vw;      /* matches [0.25, 0.5, 0.25] grid */
      width: 50vw;
      top: 52%;
      transform: translateY(-50%);
      z-index: 2;
      padding: 0 12px;
  }
  .fs-title {
      text-align:center;
      font-size: 2rem;
      font-weight:800;
      margin: 0 0 1rem 0;
      letter-spacing:.2px;
  }
  .center-chart [data-testid="stPlotlyChart"],
  .center-chart .js-plotly-plot,
  .center-chart [data-testid="stVegaLiteChart"] {
      margin: 0 auto !important;
      display: block;
  }
  .stSelectbox, .stTextInput, .stRadio, .stButton { margin-bottom: 0.45rem !important; }
  .stProgress { margin: 6px 0 12px 0 !important; }
  .stProgress div div { height: 10px !important; }
  .news-scroll {
      height: 46.5vh;
      overflow: auto;
      padding-right: 8px;
      scroll-behavior: smooth;
  }
  .disclaimer-fixed {
      position: fixed;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      width: 60%;
      text-align: center;
      font-size: 0.74rem;
      line-height: 1.35;
      color: #7b7b7b;
      background: rgba(0,0,0,0.02);
      border-radius: 8px;
      padding: 6px 14px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      backdrop-filter: blur(3px);
      z-index: 99;
  }
</style>
""", unsafe_allow_html=True)


# Project Page Layout 
left, mid, right = st.columns([0.25, 0.50, 0.25])

# Left Side
with left:
    ticker = st.text_input("Ticker", value="MSFT").strip().upper()
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"], index=0)
    st.markdown("**Sentiment engine**")
    engine_choice = st.radio("", ["VADER (fast)", "FinBERT (finance ML)"], index=0,
                             horizontal=True, label_visibility="collapsed")
    load = st.button("Load", type="primary", use_container_width=True)
    success_placeholder = st.empty()

# Middle Side 
with mid:
    st.markdown("<div class='middle-fixed'>", unsafe_allow_html=True)
    st.markdown("<div class='fs-title'>FinSignal</div>", unsafe_allow_html=True)
    st.markdown("### Historical Price Chart")
    prices = None
    display_name = None
    if load and ticker:
        try:
            prices = get_prices(ticker, period=period, interval=interval)
            if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
                st.warning("No price data returned.")
            else:
                display_name = resolve_company_name(ticker)
                header = f"{display_name} ({ticker}) — Close Price" if display_name and display_name != ticker else f"{ticker} — Close Price"
                st.markdown(f"**{header}**")
                st.markdown("<div class='center-chart'>", unsafe_allow_html=True)
                if chart_type == "Line":
                    line_chart(prices, ticker, y="close", use_container_width=True)
                else:
                    candlestick_chart(prices, ticker, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("Close price (auto-adjusted); SMA20 & SMA50 overlay for signal context.")
                success_placeholder.success(f"{ticker} loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load prices for {ticker}: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# News Fetch and Sentiment
news: List[Dict] = []
company_name = None
if load and ticker:
    try:
        company_name = resolve_company_name(ticker)
        raw_news = get_news_newsapi(
            ticker=ticker,
            company_name=company_name or ticker,
            max_items=25,
            days=7
        )
        news = normalize_news(raw_news)
        news = filter_news_for_company(news, company_name or ticker, ticker)
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        news = []

use_finbert = engine_choice.startswith("FinBERT")
titles = [n.get("title", "") for n in news]
labels: List[str] = []
scores: List[float] = []

try:
    if titles:
        res = score_headlines_finbert(titles) if use_finbert else score_headlines_vader(titles)
        if isinstance(res, tuple):
            labels, scores = res
        elif hasattr(res, "to_dict"):
            if "label" in res.columns:
                labels = [str(x).lower() for x in res["label"].tolist()]
            if "compound" in res.columns:
                scores = [float(x) for x in res["compound"].tolist()]
except Exception as e:
    st.error(f"Sentiment scoring failed: {e}")
    labels, scores = [], []

summary = summarize_label_list(labels)
net_sent = summary["net"]
sent_bar_val = (net_sent + 1.0) / 2.0
sent_color = sentiment_color_from_net(net_sent)
engine_note = "FinBERT (finance-domain transformer)" if use_finbert else "VADER (rule-based, fast)"

# Left Signal
with left:
    st.markdown("### Signal")
    sig, confidence, details = "HOLD", 0.0, {"blend_score": 0.0, "tech": {}, "news": {}}
    if isinstance(prices, pd.DataFrame) and not prices.empty:
        prices = _flatten_cols(prices)
        if "close" in prices.columns:
            close = prices["close"].dropna()
            sma20 = close.rolling(20).mean()
            sma50 = close.rolling(50).mean()
            momentum = (close.iloc[-1] / close.iloc[-10] - 1.0) if len(close) >= 10 else 0.0
            sma20_slope = (sma20.iloc[-1] - sma20.iloc[-5]) / 5 if len(sma20.dropna()) >= 5 else 0.0
            sma50_slope = (sma50.iloc[-1] - sma50.iloc[-5]) / 5 if len(sma50.dropna()) >= 5 else 0.0
            sma_cross = 1 if (len(sma20.dropna()) and len(sma50.dropna()) and sma20.iloc[-1] > sma50.iloc[-1]) else -1

            tech_component = (
                0.4 * (1 if sma_cross > 0 else -1) +
                0.3 * max(-1.0, min(1.0, sma20_slope * 50)) +
                0.3 * max(-1.0, min(1.0, momentum * 10))
            )
            blend_score = 0.6 * tech_component + 0.4 * net_sent
            sig = "BUY" if blend_score >= 0.15 else ("SELL" if blend_score <= -0.15 else "HOLD")
            confidence = min(1.0, abs(blend_score) * 1.5)
            details = {"blend_score": float((blend_score + 1) / 2), "tech": {}, "news": {"net_score": net_sent, "counts": summary}}
    render_signal_panel(sig, confidence, details, sentiment_note=engine_note)

# Right Side: News Sentiment Layout + Scroll Recent News ("playlist" format)
with right:
    st.markdown("### News Sentiment")
    header_html, _ = news_sentiment_panel(labels)
    st.markdown(header_html, unsafe_allow_html=True)
    render_colored_bar(sent_bar_val, sent_color, "Sentiment gauge (red ↔ neutral ↔ green)")
    st.caption(f"Sentiment engine: {engine_note}.")
    st.markdown("---")
    st.markdown("### Recent News")

    items_html = []
    if news:
        for idx, item in enumerate(news):
            title = (item.get("title") or "").strip()
            src = item.get("source") or ""
            t = item.get("published_at") or ""
            url = item.get("url") or ""
            lab = labels[idx] if idx < len(labels) else "Neutral"
            badge_html = sentiment_badge(lab)
            meta_bits = [b for b in [src, str(t)] if b]
            meta_line = " · ".join(meta_bits)
            link_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title
            items_html.append(
                f"<div style='margin-bottom:.65rem;'>{badge_html}&nbsp;"
                f"<span style='opacity:.85;'>{meta_line}</span><br/>{link_html}</div>"
            )
    else:
        items_html.append("<div style='opacity:.75;'>No recent, relevant headlines found.</div>")

    st.markdown(f"""
    <div class="news-scroll" id="recent-news">
      {''.join(items_html)}
    </div>
    """, unsafe_allow_html=True)

# FinSignal Disclaimer 
st.markdown("""
<div class="disclaimer-fixed">
  Disclaimer: FinSight is created and maintained by Utsha Saha. The information and analyses provided by FinSight are for educational and informational purposes only and do not constitute financial, investment, or trading advice. All investment decisions should be made independently and at your own discretion. Utsha Saha and FinSight assume no responsibility or liability for any financial losses or outcomes resulting from the use of this application.
</div>
""", unsafe_allow_html=True)
