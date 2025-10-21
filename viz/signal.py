# Imports Streamlit
import streamlit as st

# Creates a HTML Badge for BUY, SELL, or HOLD 
def _pill(label: str, color: str, text_color: str = "white"):
    return f"""
        <span style="
            display:inline-block;
            padding:3px 8px;
            border-radius:999px;
            background:{color};
            color:{text_color};
            font-size:11px;
            font-weight:700;">
            {label}
        </span>
    """

# Builds and displays Signal Summary Section
def render_signal_panel(signal: str, confidence: float, details: dict, sentiment_note: str = ""):
    # Signal and Color Codes
    if signal == "BUY":
        pill = _pill("BUY", "#16a34a")
    elif signal == "SELL":
        pill = _pill("SELL", "#ef4444")
    else:
        pill = _pill("HOLD", "#6b7280")

    conf_txt = "High" if confidence >= 0.66 else ("Medium" if confidence >= 0.33 else "Low")
    header = (
        f'{pill}&nbsp;&nbsp;<span style="opacity:.9; font-size:13px;">'
        f'Confidence: <strong>{conf_txt}</strong></span>'
    )
    st.markdown(header, unsafe_allow_html=True)

    pct = int(min(max(details.get("blend_score", 0.0) * 100, 0), 100))
    st.progress(pct, text=f"Composite score: +{pct} / 100")

    content = (
        "Blend: 60% technical • 40% sentiment (news)\n\n"
        "**Technical details**  \n"
        f"- SMA20 vs SMA50 cross: **{details.get('tech', {}).get('sma20_vs_50', 0):+d}**  \n"
        f"- 5–20d momentum: **{details.get('tech', {}).get('momentum', 0.0):+.2f}**  \n"
        f"- SMA20 slope: **{details.get('tech', {}).get('sma20_slope', 0.0):+.2f}**  \n"
        f"- SMA50 slope: **{details.get('tech', {}).get('sma50_slope', 0.0):+.2f}**\n\n"
        "**Sentiment**  \n"
        f"- Net news score: **{details.get('news', {}).get('net_score', 0.0):+.2f}**  \n"
        f"- Counts (P/N/N): **{details.get('news', {}).get('counts', {}).get('pos',0)} / "
        f"{details.get('news', {}).get('counts', {}).get('neu',0)} / "
        f"{details.get('news', {}).get('counts', {}).get('neg',0)}**"
    )

    try:
        with st.popover("Details", use_container_width=True):
            st.caption(content)
            if sentiment_note:
                st.caption(sentiment_note)
    except Exception:
        with st.expander("Details", expanded=False):
            st.caption(content)
            if sentiment_note:
                st.caption(sentiment_note)
