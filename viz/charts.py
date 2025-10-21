import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Function to ensure DataFrame columns are lowercase and flattened
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df = df.rename(columns=lambda c: str(c).lower())
    return df

# Function to plot line chart of stock prices with moving averages (SMA20 and SMA50)
def line_chart(prices: pd.DataFrame, ticker: str, y: str = "close", use_container_width: bool = True):
    prices = _normalize_cols(prices)
    if y not in prices.columns:
        raise ValueError(f"'{y}' column not found. Available: {list(prices.columns)}")

    df = prices.copy()
    df["sma20"] = df[y].rolling(20).mean()
    df["sma50"] = df[y].rolling(50).mean()

    # no figure title; we render subtitle in app.py to avoid legend overlap
    fig = px.line(df, x=df.index, y=y)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], mode="lines", name="SMA50"))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),  # small top margin since there is no title
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,            # sits just above the plot area
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=use_container_width)

# Function to plot candlestick chart of stock prices (open-high-low-close) with moving averages (SMA20 and SMA50)
def candlestick_chart(prices: pd.DataFrame, ticker: str, use_container_width: bool = True):
    prices = _normalize_cols(prices)
    needed = {"open","high","low","close"}
    missing = needed - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns for candlestick: {missing}")

    df = prices.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name=ticker
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma20"], mode="lines", name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], mode="lines", name="SMA50"))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig, use_container_width=use_container_width)