# Imports of Pandas and Yahoo! Finannce
import yfinance as yf
import pandas as pd

# Gets the price for stock ticker
def get_prices_yf(ticker: str, period="6mo", interval="1d") -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",   # << force flat columns
    )
    return df
