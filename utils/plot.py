# Imports
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Line Chart Generation for Selected Price Column 
def line_chart(prices: pd.DataFrame, ticker: str, y="close", use_container_width=True):
    fig = px.line(prices, x=prices.index, y=y, title=f"{ticker} — Close Price")
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(y=0.93, x=0.02, xanchor="left", yanchor="top"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(0,0,0,0)"
        )
    )
    st.plotly_chart(fig, use_container_width=use_container_width)
