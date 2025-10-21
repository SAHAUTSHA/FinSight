
import time
import pandas as pd
import yfinance as yf

def get_prices_yf(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Download historical price data for a given ticker using yfinance.
    Includes retry logic, error handling, and flat column formatting.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. 'MSFT')
    period : str, optional
        Time period for data (default '6mo')
    interval : str, optional
        Candle interval (default '1d')

    Returns
    -------
    pd.DataFrame
        Price data with columns: open, high, low, close, adj close, volume
        Empty DataFrame if download fails or is rate-limited.
    """
    tries = 3
    for attempt in range(tries):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="column",    # flatten MultiIndex
                threads=False,         # avoid Yahoo throttling
                prepost=False,
            )

            # Ensure valid DataFrame
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.columns = [c.lower() for c in df.columns]  # normalize
                return df

        except Exception as e:
            print(f"[WARN] Attempt {attempt+1} failed for {ticker}: {e}")

        # backoff before retry
        time.sleep(1.5 * (attempt + 1))

    # Return an empty DataFrame on failure
    print(f"[ERROR] All attempts failed for {ticker}. Returning empty DataFrame.")
    return pd.DataFrame(columns=["open", "high", "low", "close", "adj close", "volume"])
