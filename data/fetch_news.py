# data/fetch_news.py

# Imports
import os
import re
import datetime as dt
import requests
import pandas as pd

# Streamlit is optional (used for secrets + UI warnings)
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # makes the module usable outside Streamlit too

# Words to ignore when building brand tokens
_STOPWORDS = {
    "inc", "corp", "corporation", "co", "plc", "ltd", "nv", "sa",
    "ag", "limited", "holdings", "company", "group"
}

# ---------- secrets / config helpers ----------

def _get_secret(name: str, default: str | None = None) -> str | None:
    """
    Resolve a secret in this order:
    1) Streamlit Cloud secrets (st.secrets)
    2) Environment variable
    3) Optionally load from .env if python-dotenv is installed locally
    """
    # 1) Streamlit secrets
    try:
        if st is not None and hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    # 2) Environment variable
    val = os.getenv(name)
    if val:
        return val

    # 3) Optional: try .env (local dev only)
    try:
        from dotenv import load_dotenv  # optional; don't require in prod
        load_dotenv()
        return os.getenv(name, default)
    except Exception:
        return default


# ---------- text helpers ----------

def _brand_tokens(company_name: str) -> list[str]:
    """Return compact token variants from a company name for better matching."""
    if not company_name:
        return []
    parts = [p for p in re.findall(r"[A-Za-z]+", company_name) if p.lower() not in _STOPWORDS]
    if not parts:
        return [company_name.strip()]
    tokens = [" ".join(parts)]
    tokens.append(parts[0])  # leading word often distinctive (e.g., "Apple" from "Apple Inc.")
    # de-dup while preserving order
    return list(dict.fromkeys(tokens))


# ---------- main API ----------

def get_news_newsapi(
    ticker: str,
    company_name: str | None = None,
    max_items: int = 5,
    days: int = 7,
) -> pd.DataFrame:
    """
    Fetch recent company-relevant headlines via NewsAPI and return a tidy DataFrame:
    columns = [published_at, source, title, url]
    """
    key = _get_secret("NEWSAPI_KEY")
    if not key:
        # Fail gracefully in the UI; don't crash the app
        if st is not None:
            st.warning("NEWSAPI_KEY is missing. Add it in Settings → Secrets or set it in the environment.")
        return pd.DataFrame(columns=["published_at", "source", "title", "url"])

    # clamp ranges to keep the query reasonable
    max_items = max(1, min(int(max_items or 5), 10))
    days = max(1, min(int(days or 7), 30))

    ticker = (ticker or "").upper().strip()
    brand_tokens = _brand_tokens(company_name or "")

    # Build a conservative query to reduce false positives
    # Use ticker only if it's at least 3 chars (avoid 'A', 'AI' collisions)
    query_parts: list[str] = []
    if len(ticker) >= 3:
        query_parts.append(f'"{ticker}"')
    for tok in brand_tokens:
        query_parts.append(f'"{tok}"')

    if not query_parts:
        # fallback if all else fails
        if ticker:
            query_parts = [f'"{ticker}"']
        elif company_name:
            query_parts = [f'"{company_name}"']
        else:
            return pd.DataFrame(columns=["published_at", "source", "title", "url"])

    query = " OR ".join(query_parts)
    since = (dt.datetime.utcnow() - dt.timedelta(days=days)).date().isoformat()

    # Filter out developer/package domains that often pollute results
    exclude = ",".join([
        "pypi.org","github.com","dev.to","medium.com",
        "npmjs.com","rubygems.org","packaging.python.org",
        "crates.io","hub.docker.com"
    ])

    params = {
        "q": query,
        "qInTitle": " OR ".join([p for p in query_parts if (" " in p) or (len(p.strip('"')) >= 3)]),
        "searchIn": "title,description",
        "from": since,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "excludeDomains": exclude,
        "apiKey": key,
    }

    headers = {
        "User-Agent": "FinSight/1.0 (news fetcher)"
    }

    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        arts = (resp.json() or {}).get("articles", []) or []
        if not arts:
            return pd.DataFrame(columns=["published_at", "source", "title", "url"])

        rows = [{
            "published_at": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name", "") or "",
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "url": a.get("url") or "",
        } for a in arts]

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["published_at", "source", "title", "url"])

        # Normalize time and build relevance filters
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        ticker_pat = re.compile(rf"\b{re.escape(ticker)}\b", re.I) if len(ticker) >= 3 else None
        brand_pats = [re.compile(rf"\b{re.escape(tok)}\b", re.I) for tok in brand_tokens]

        def _is_relevant(title: str, desc: str) -> bool:
            # strong signal if ticker or brand present in title; allow desc as a backup
            if ticker_pat and ticker_pat.search(title):
                return True
            if any(p.search(title) for p in brand_pats):
                return True
            if ticker_pat and ticker_pat.search(desc):
                return True
            if any(p.search(desc) for p in brand_pats):
                return True
            return False

        df = df[df.apply(lambda r: _is_relevant(r.get("title", ""), (r.get("description") or "")[:160]), axis=1)]
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
        # Quiet failure → empty frame; keeps the app responsive
        return pd.DataFrame(columns=["published_at", "source", "title", "url"])
