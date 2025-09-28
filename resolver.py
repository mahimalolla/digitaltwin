from typing import List, Optional, Dict
import requests

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

def _yahoo_search(query: str, quotes_count: int = 6) -> List[Dict]:
    """
    Calls Yahoo Finance's suggestion API (no API key).
    Returns a list of quote dicts with fields like symbol, shortname, exchDisp, typeDisp, etc.
    """
    params = {
        "q": query.strip(),
        "quotesCount": quotes_count,
        "newsCount": 0,
        "listsCount": 0,
        "quotesQueryId": "tss_match_phrase_query",
        "enableCb": True
    }
    r = requests.get(YAHOO_SEARCH_URL, params=params, timeout=8)
    r.raise_for_status()
    data = r.json() or {}
    return data.get("quotes", [])

def resolve_company(query: str) -> Optional[Dict]:
    """
    Accepts either a company name ('Apple', 'Reliance') or ticker ('AAPL', 'RELIANCE.NS').
    Returns: {'ticker': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ'} or None.
    """
    if not query or not query.strip():
        return None

    candidates = _yahoo_search(query, quotes_count=10)
    if not candidates:
        return None

    # allow equities / funds; ignore FX/indices
    equities = [
        c for c in candidates
        if c.get("symbol") and (c.get("typeDisp") or "").lower() in {"equity", "etf", "mutual fund", "fund"}
    ]
    if not equities:
        return None

    upper_query = query.strip().upper()
    # exact ticker match
    for c in equities:
        if c["symbol"].upper() == upper_query:
            return {
                "ticker": c["symbol"],
                "name": c.get("shortname") or c.get("longname") or c["symbol"],
                "exchange": c.get("exchDisp") or ""
            }

    # score by name overlap + known exchanges
    q_lower = query.strip().lower()
    q_tokens = set(q_lower.split())
    scored = []
    for c in equities:
        name = (c.get("shortname") or c.get("longname") or "").lower()
        score = 0
        if q_lower in name:
            score += 5
        score += len(q_tokens & set(name.split()))
        exch = (c.get("exchDisp") or "").upper()
        if any(x in exch for x in ["NASDAQ", "NYSE", "NSE", "BSE", "LSE", "ASX", "TSX", "HKEX"]):
            score += 2
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = (scored[0][1] if scored else equities[0])

    return {
        "ticker": best["symbol"],
        "name": best.get("shortname") or best.get("longname") or best["symbol"],
        "exchange": best.get("exchDisp") or ""
    }
