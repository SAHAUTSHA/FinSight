# Imports
import os
import re
import datetime as dt
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Expects NEWSAPI_KEY in .env 

_STOPWORDS = {"inc", "corp", "corporation", "co", "plc", "ltd", "nv", "sa", "ag", "limited", "holdings"}

# Generates a list of token variations for Company Names 
def _brand_tokens(company_name: str) -> list[str]:
    if not company_name:
        return []
    parts = [p for p in re.findall(r"[A-Za-z]+", company_name) if p.lower() not in _STOPWORDS]
    if not parts:
        return [company_name.strip()]
    tokens = [" ".join(parts)]
    if parts:
        tokens.append(parts[0])
    return list(dict.fromkeys(tokens))

# Fetches the latest company related news using NewsAPI, filters relevance, and returns pandas DataFrame
def get_news_newsapi(
    ticker: str,
    company_name: str | None = None,
    max_items: int = 5,
    days: int = 7,
) -> pd.DataFrame:
    key = os.getenv("NEWSAPI_KEY")
    if not key:
        return pd.DataFrame(columns=["published_at", "source", "title", "url"])

    ticker = (ticker or "").upper().strip()
    brand_tokens = _brand_tokens(company_name or "")

    query_parts = []
    if len(ticker) >= 3:
        query_parts.append(f'"{ticker}"')
    for tok in brand_tokens:
        query_parts.append(f'"{tok}"')
    if not query_parts:
        query_parts = [f'"{ticker}"'] if ticker else [f'"{company_name}"']

    query = " OR ".join(query_parts)
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).date().isoformat()

    exclude = ",".join([
        "pypi.org","github.com","dev.to","medium.com",
        "npmjs.com","rubygems.org","packaging.python.org",
        "crates.io","hub.docker.com"
    ])

    params = {
        "q": query,
        "qInTitle": " OR ".join([p for p in query_parts if " " in p or len(p) >= 3]),
        "searchIn": "title,description",
        "from": since,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "excludeDomains": exclude,
        "apiKey": key,
    }

    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        resp.raise_for_status()
        arts = resp.json().get("articles", [])
        if not arts:
            return pd.DataFrame(columns=["published_at", "source", "title", "url"])

        rows = [{
            "published_at": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name", ""),
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "url": a.get("url") or "",
        } for a in arts]
        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        ticker_pat = re.compile(rf"\b{re.escape(ticker)}\b", re.I) if len(ticker) >= 3 else None
        brand_pats = [re.compile(rf"\b{re.escape(tok)}\b", re.I) for tok in brand_tokens]

        def is_strong(row) -> bool:
            title = row["title"] or ""
            desc = (row["description"] or "")[:120]
            if ticker_pat and ticker_pat.search(title):
                return True
            if any(p.search(title) for p in brand_pats):
                return True
            if ticker_pat and ticker_pat.search(desc):
                return True
            if any(p.search(desc) for p in brand_pats):
                return True
            return False

        df = df[df.apply(is_strong, axis=1)]
        if df.empty:
            return pd.DataFrame(columns=["published_at", "source", "title", "url"])

        df = (
            df.drop_duplicates(subset=["title"])
              .sort_values("published_at", ascending=False)
              .head(max_items)
              .reset_index(drop=True)
        )
        return df[["published_at", "source", "title", "url"]]

    except Exception:
        return pd.DataFrame(columns=["published_at", "source", "title", "url"])
