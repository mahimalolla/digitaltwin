# search_api.py
# -----------------------------------------
# Warren Buffett Digital Twin - Backend API (demo-friendly)
# -----------------------------------------
# Endpoints:
#   GET  /health, /healthz
#   POST /search
#   GET  /predict?ticker=AAPL
#   GET  /predict?company=Apple
#   GET  /autocomplete?q=app
#
# Notes:
# - Name->ticker resolution prefers an OFFLINE map first (so demos work without internet).
# - Fundamentals: try yfinance; if it fails and DEMO_MODE=True, fall back to demo numbers.
# - Price history: try yfinance; else generate a synthetic sparkline.

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json, torch, math, random, time

# ---------- RAG / LLM ----------
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama

# ---------- Fundamentals ----------
import yfinance as yf
import numpy as np
import pandas as pd

# ---------- Name->Ticker (online optional) ----------
import requests
from requests import RequestException

# ====================================================
# Demo toggles & offline data
# ====================================================
DEMO_MODE = True                 # if yfinance fails, show demo fundamentals
DEMO_PRICE_POINTS = 180          # synthetic series length if live history fails

# Offline NAME -> TICKER map (used FIRST)
OFFLINE_NAME_TO_TICKER = {
    # US
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META", "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
    # India
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
}

# Demo fundamentals snapshot (used only when live fetch fails)
DEMO_FUNDAMENTALS = {
    "AAPL": {"company": "Apple Inc.", "price": 220.00, "eps": 6.40, "pe": 34.4, "roe": 0.15, "debtToEquity": 1.60, "earningsGrowth": 0.07},
    "MSFT": {"company": "Microsoft Corporation", "price": 430.00, "eps": 11.50, "pe": 37.4, "roe": 0.40, "debtToEquity": 0.45, "earningsGrowth": 0.12},
    "GOOGL": {"company": "Alphabet Inc.", "price": 165.00, "eps": 6.80, "pe": 24.3, "roe": 0.30, "debtToEquity": 0.10, "earningsGrowth": 0.11},
    "AMZN": {"company": "Amazon.com, Inc.", "price": 180.00, "eps": 3.60, "pe": 50.0, "roe": 0.20, "debtToEquity": 0.75, "earningsGrowth": 0.18},
    "META": {"company": "Meta Platforms, Inc.", "price": 480.00, "eps": 17.50, "pe": 27.4, "roe": 0.32, "debtToEquity": 0.09, "earningsGrowth": 0.16},
    "TSLA": {"company": "Tesla, Inc.", "price": 250.00, "eps": 3.20, "pe": 78.1, "roe": 0.14, "debtToEquity": 0.90, "earningsGrowth": 0.12},
    "NVDA": {"company": "NVIDIA Corporation", "price": 1150.00, "eps": 25.00, "pe": 46.0, "roe": 0.48, "debtToEquity": 0.60, "earningsGrowth": 0.35},
    "BRK-B": {"company": "Berkshire Hathaway Inc. (Class B)", "price": 420.00, "eps": 18.00, "pe": 23.3, "roe": 0.11, "debtToEquity": 0.28, "earningsGrowth": 0.08},
    "RELIANCE.NS": {"company": "Reliance Industries Ltd", "price": 3100.00, "eps": 115.00, "pe": 27.0, "roe": 0.12, "debtToEquity": 0.80, "earningsGrowth": 0.09},
    "TCS.NS": {"company": "Tata Consultancy Services", "price": 4000.00, "eps": 145.00, "pe": 27.6, "roe": 0.43, "debtToEquity": 0.05, "earningsGrowth": 0.10},
    "INFY.NS": {"company": "Infosys Ltd", "price": 1600.00, "eps": 61.00, "pe": 26.2, "roe": 0.32, "debtToEquity": 0.05, "earningsGrowth": 0.09},
    "HDFCBANK.NS": {"company": "HDFC Bank Ltd", "price": 1600.00, "eps": 82.00, "pe": 19.5, "roe": 0.16, "debtToEquity": 0.00, "earningsGrowth": 0.12},
}

# ====================================================
# FastAPI app
# ====================================================
app = FastAPI(title="Investment Digital Twin API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
@app.get("/healthz")
def health():
    return {"ok": True}

# ====================================================
# RAG pieces
# ====================================================
model = SentenceTransformer("all-MiniLM-L6-v2")

docs: List[dict] = []
embeddings: List[List[float]] = []
emb_path = Path("data/final/embeddings.jsonl")
if emb_path.exists() and emb_path.stat().st_size > 0:
    with emb_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj)
            embeddings.append(obj["embedding"])
    corpus_embeddings = torch.tensor(embeddings) if embeddings else None
else:
    corpus_embeddings = None

LLM_PATH = "model/llms/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
CTX_SIZE = 2048
MAX_NEW_TOKENS = 256
TOKEN_CHAR_RATIO = 4.0
SAFETY_MARGIN = 64

_llm = None
_llm_err = None
try:
    _llm = Llama(model_path=LLM_PATH, n_ctx=CTX_SIZE, n_threads=4, n_gpu_layers=0, verbose=False)
except Exception as e:
    _llm_err = str(e)

class QueryIn(BaseModel):
    query: str = Field(..., description="Natural-language question")

def _trim_context_for_ctx(query: str, context: str) -> str:
    base = ("You are Warren Buffett's financial digital twin.\n\n"
            "Use the following context to answer concisely and cite key ideas when helpful.\n\n"
            "Context:\n\n\n"
            f"Question:\n{query}\n\n"
            "Answer:")
    approx_base_tokens = len(base) / TOKEN_CHAR_RATIO
    allowed_tokens = CTX_SIZE - MAX_NEW_TOKENS - SAFETY_MARGIN
    allowed_ctx_tokens = max(0, int(allowed_tokens - approx_base_tokens))
    allowed_ctx_chars = int(allowed_ctx_tokens * TOKEN_CHAR_RATIO)
    return context[: max(0, allowed_ctx_chars)]

def generate_answer(query: str, context: str) -> str:
    if _llm is None:
        raise HTTPException(status_code=503, detail=f"LLM not available: {LLM_PATH} ({_llm_err or 'init error'})")
    trimmed = _trim_context_for_ctx(query, context)
    prompt = f"""You are Warren Buffett's financial digital twin.

Use the following context to answer concisely and cite key ideas when helpful.

Context:
{trimmed}

Question:
{query}

Answer:"""
    try:
        out = _llm(prompt, max_tokens=MAX_NEW_TOKENS, temperature=0.2, stop=["</s>"])
        return out["choices"][0]["text"].strip()
    except Exception:
        trimmed = trimmed[: max(0, int(len(trimmed) * 0.6))]
        out = _llm(prompt, max_tokens=min(128, MAX_NEW_TOKENS), temperature=0.2, stop=["</s>"])
        return out["choices"][0]["text"].strip()

@app.post("/search")
def search_chunks(query: QueryIn):
    if not query.query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    if corpus_embeddings is None or not docs:
        raise HTTPException(status_code=503, detail="RAG corpus not loaded. Build data/final/embeddings.jsonl first.")
    q_emb = model.encode(query.query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, corpus_embeddings, top_k=5)[0]
    top_results = []
    for h in hits:
        idx = int(h["corpus_id"])
        doc = docs[idx] if 0 <= idx < len(docs) else {"text": "", "source": ""}
        top_results.append({"score": round(float(h.get("score", 0.0)), 4), "source": doc.get("source", ""), "text": doc.get("text", "")})
    context = "\n".join(d["text"] for d in top_results if d["text"])
    answer = generate_answer(query.query, context)
    return {"query": query.query, "answer": answer, "results": top_results}

# ====================================================
# Predictor helpers
# ====================================================
def _safe_get(series_or_df, key):
    try:
        return float(series_or_df.loc[key].iloc[0])
    except Exception:
        return None

def _graham_intrinsic(eps: Optional[float], growth: Optional[float]) -> Optional[float]:
    if eps is None:
        return None
    g_pct = (growth or 0.08) * 100
    Y = 4.4
    return eps * (8.5 + 2 * g_pct) * 4.4 / Y

def score_buffett(m: dict):
    reasons, score = [], 0
    roe, pe, dte, g, price, iv = m["roe"], m["pe"], m["debtToEquity"], m["earningsGrowth"], m["price"], m["intrinsicValue"]
    if roe is not None:
        if roe >= 0.15: score += 2; reasons.append(f"ROE {roe:.1%} >= 15% (strong)")
        elif roe >= 0.10: score += 1; reasons.append(f"ROE {roe:.1%} >= 10% (ok)")
        else: reasons.append(f"ROE {roe:.1%} < 10% (weak)")
    if pe is not None:
        if pe <= 20: score += 2; reasons.append(f"P/E {pe:.1f} <= 20 (reasonable)")
        elif pe <= 25: score += 1; reasons.append(f"P/E {pe:.1f} <= 25 (borderline)")
        else: reasons.append(f"P/E {pe:.1f} > 25 (expensive)")
    if dte is not None:
        if dte <= 0.5: score += 2; reasons.append(f"Debt/Equity {dte:.2f} <= 0.5 (conservative)")
        elif dte <= 1.0: score += 1; reasons.append(f"Debt/Equity {dte:.2f} <= 1.0 (moderate)")
        else: reasons.append(f"Debt/Equity {dte:.2f} > 1.0 (high)")
    if g is not None:
        if g >= 0.10: score += 2; reasons.append(f"Earnings growth {g:.1%} >= 10% (healthy)")
        elif g >= 0.05: score += 1; reasons.append(f"Earnings growth {g:.1%} >= 5% (ok)")
        else: reasons.append(f"Earnings growth {g:.1%} < 5% (low)")
    if iv is not None and price is not None and price != 0:
        margin = (iv - price) / price
        if margin >= 0.25: score += 2; reasons.append(f"Price below intrinsic by {margin:.0%} (value)")
        elif margin >= 0.0: score += 1; reasons.append("Near intrinsic (fair)")
        else: reasons.append(f"Above intrinsic by {abs(margin):.0%} (premium)")
    if score >= 8: label, color = "Likely Buy", "green"
    elif score >= 5: label, color = "Hold / Watchlist", "yellow"
    else: label, color = "Avoid / Too Expensive", "red"
    return {"score": score, "label": label, "color": color, "reasons": reasons}

# ====================================================
# Live fundamentals (yfinance) + demo fallback
# ====================================================
def compute_metrics_yfinance(ticker: str) -> Optional[Dict]:
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.get_info()
        except Exception:
            try:
                info = tk.info
            except Exception:
                info = {}
        company = info.get("longName") or info.get("shortName") or ticker.upper()
        try:
            inc = tk.financials
            bal = tk.balance_sheet
        except Exception:
            inc = pd.DataFrame(); bal = pd.DataFrame()
        net_income = _safe_get(inc, "Net Income")
        equity = _safe_get(bal, "Total Stockholder Equity") or _safe_get(bal, "Total Equity Gross Minority Interest")
        total_liab = _safe_get(bal, "Total Liab")
        fast = getattr(tk, "fast_info", {}) or {}
        shares = None
        for k in ("sharesOutstanding", "shares_outstanding"):
            v = fast.get(k)
            if v:
                try:
                    shares = float(v); break
                except Exception:
                    pass
        if shares is None:
            try:
                shares = float(info.get("sharesOutstanding")) if info.get("sharesOutstanding") else None
            except Exception:
                shares = None
        price = None
        try:
            price = float(fast.get("lastPrice") or fast.get("last_price") or info.get("currentPrice"))
        except Exception:
            price = None
        eps = None
        try:
            eps = float(info.get("trailingEps")) if info.get("trailingEps") is not None else None
        except Exception:
            eps = None
        if eps is None and net_income and shares:
            eps = net_income / shares
        pe = None
        for k in ("peRatio", "trailingPE"):
            try:
                val = fast.get(k) if k in fast else info.get(k)
                if val:
                    pe = float(val); break
            except Exception:
                pass
        if pe is None and price and eps and eps != 0:
            pe = price / eps
        roe = None
        if net_income is not None and equity not in (None, 0):
            roe = net_income / equity
        d_to_e = None
        if total_liab is not None and equity not in (None, 0):
            d_to_e = total_liab / equity
        growth = None
        try:
            if not inc.empty and "Net Income" in inc.index and inc.shape[1] >= 2:
                ni = inc.loc["Net Income"].dropna().astype(float).sort_index()
                if len(ni) >= 2:
                    n_years = max(1, len(ni) - 1)
                    start, end = float(ni.iloc[0]), float(ni.iloc[-1])
                    if start > 0 and end > 0:
                        growth = (end / start) ** (1 / n_years) - 1
        except Exception:
            pass
        intrinsic = _graham_intrinsic(eps, growth)
        return {
            "company": company,
            "ticker": ticker.upper(),
            "metrics": {
                "roe": roe, "pe": pe, "debtToEquity": d_to_e, "earningsGrowth": growth,
                "eps": eps, "price": price, "intrinsicValue": intrinsic
            }
        }
    except Exception:
        return None

def compute_metrics_demo(tkr: str) -> Optional[Dict]:
    sym = tkr.upper()
    snap = DEMO_FUNDAMENTALS.get(sym)
    if not snap:
        return None
    eps = snap["eps"]; growth = snap["earningsGrowth"]
    intrinsic = _graham_intrinsic(eps, growth)
    return {
        "company": snap["company"],
        "ticker": sym,
        "metrics": {
            "roe": snap["roe"],
            "pe": snap["pe"],
            "debtToEquity": snap["debtToEquity"],
            "earningsGrowth": growth,
            "eps": eps,
            "price": snap["price"],
            "intrinsicValue": intrinsic
        }
    }

# ====================================================
# Resolver (name -> ticker)
# ====================================================
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

def _yahoo_search(query: str, quotes_count: int = 8) -> List[dict]:
    try:
        params = {"q": query.strip(), "quotesCount": quotes_count, "newsCount": 0, "listsCount": 0,
                  "quotesQueryId": "tss_match_phrase_query", "enableCb": True}
        r = requests.get(YAHOO_SEARCH_URL, params=params, timeout=5)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("quotes", [])
    except RequestException:
        return []

def resolve_company(query: str) -> Optional[Dict]:
    """Prefer offline map. If not found, try Yahoo. Strict ticker passthrough for ALL-CAPS."""
    if not query or not query.strip():
        return None
    q = query.strip()
    # strict ticker passthrough (e.g., AAPL, RELIANCE.NS)
    if q == q.upper() and " " not in q and 1 <= len(q) <= 12 and q.replace('.', '').replace('-', '').isalnum():
        return {"ticker": q, "name": q, "exchange": ""}

    # offline map first (demo-friendly)
    key = q.lower().strip()
    if key in OFFLINE_NAME_TO_TICKER:
        sym = OFFLINE_NAME_TO_TICKER[key]
        return {"ticker": sym, "name": q.title(), "exchange": ""}

    # try Yahoo if available
    quotes = _yahoo_search(q, quotes_count=10)
    if not quotes:
        return None
    equities = [c for c in quotes if c.get("symbol") and (c.get("typeDisp") or "").lower() in {"equity", "etf", "mutual fund", "fund"}]
    if not equities:
        return None
    q_tokens = set(key.split())
    scored: List[Tuple[int, dict]] = []
    for c in equities:
        name = (c.get("shortname") or c.get("longname") or "").lower()
        score = 0
        if key in name: score += 5
        score += len(q_tokens & set(name.split()))
        exch = (c.get("exchDisp") or "").upper()
        if any(x in exch for x in ["NASDAQ", "NYSE", "NSE", "BSE", "LSE", "ASX", "TSX", "HKEX"]):
            score += 2
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    best = (scored[0][1] if scored else equities[0])
    return {"ticker": best["symbol"], "name": best.get("shortname") or best.get("longname") or best["symbol"], "exchange": best.get("exchDisp") or ""}

# ====================================================
# Autocomplete (uses Yahoo if available; falls back to offline map)
# ====================================================
class AutocompleteItem(BaseModel):
    ticker: str
    name: str
    exchange: str
    type: str

@app.get("/autocomplete", response_model=List[AutocompleteItem])
def autocomplete(q: str = Query(..., min_length=1)):
    quotes = _yahoo_search(q, quotes_count=6)
    items: List[AutocompleteItem] = []
    if quotes:
        for c in quotes:
            sym = c.get("symbol")
            if not sym: continue
            items.append(AutocompleteItem(
                ticker=sym,
                name=c.get("shortname") or c.get("longname") or sym,
                exchange=c.get("exchDisp") or "",
                type=c.get("typeDisp") or ""
            ))
        return items
    # offline fallback
    ql = q.lower()
    for name, sym in OFFLINE_NAME_TO_TICKER.items():
        if ql in name:
            items.append(AutocompleteItem(ticker=sym, name=name.title(), exchange="", type="Equity"))
    return items[:6]

# ====================================================
# Predict endpoint
# ====================================================
@app.get("/predict")
def predict(
    ticker: Optional[str] = Query(None, min_length=1, description="If provided, used directly"),
    company: Optional[str] = Query(None, min_length=1, description="Human name, e.g., 'Apple' or 'Reliance'")
):
    # Resolve -> final ticker
    if ticker:
        final_ticker = ticker.strip().upper()
        resolved_meta = {"ticker": final_ticker, "name": final_ticker, "exchange": ""}
    elif company:
        resolved = resolve_company(company.strip())
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Could not resolve company: {company}")
        final_ticker = resolved["ticker"].upper()
        resolved_meta = resolved
    else:
        raise HTTPException(status_code=400, detail="Provide either 'ticker' or 'company'.")

    # Try live fundamentals
    data = compute_metrics_yfinance(final_ticker)

    # Demo fallback if needed
    if data is None and DEMO_MODE:
        data = compute_metrics_demo(final_ticker)

    if data is None:
        raise HTTPException(status_code=502, detail=f"Failed to fetch fundamentals for {final_ticker}")

    metrics = data["metrics"]
    verdict = score_buffett(metrics)

    # Price history: try live first; else synthetic line
    price_history: List[float] = []
    try:
        tk = yf.Ticker(final_ticker)
        hist = tk.history(period="6mo", interval="1d")
        if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
            price_history = [float(x) for x in hist["Close"].fillna(method="ffill").tail(DEMO_PRICE_POINTS).tolist()]
    except Exception:
        price_history = []

    if not price_history:
        # synthetic series around current price for demo visuals
        base = metrics.get("price") or 100.0
        rng = random.Random(42)
        series = []
        level = base * 0.9
        for i in range(DEMO_PRICE_POINTS):
            drift = 1 + 0.0008 * math.sin(i/11.0)  # gentle trend
            noise = 1 + rng.uniform(-0.004, 0.004)
            level = level * drift * noise
            series.append(round(level, 2))
        price_history = series

    # display-friendly fields
    margin_pct = None
    if metrics.get("intrinsicValue") is not None and metrics.get("price") not in (None, 0):
        margin_pct = (metrics["intrinsicValue"] - metrics["price"]) / metrics["price"]

    disp = {
        "roe": f"{metrics['roe']*100:.1f}%" if metrics['roe'] is not None else "—",
        "pe": f"{metrics['pe']:.1f}" if metrics['pe'] is not None else "—",
        "debtToEquity": f"{metrics['debtToEquity']:.2f}" if metrics['debtToEquity'] is not None else "—",
        "earningsGrowth": f"{metrics['earningsGrowth']*100:.1f}%" if metrics['earningsGrowth'] is not None else "—",
        "intrinsicValue": f"${metrics['intrinsicValue']:.2f}" if metrics['intrinsicValue'] is not None else "—",
        "price": f"${metrics['price']:.2f}" if metrics['price'] is not None else "—",
        "eps": f"{metrics['eps']:.2f}" if metrics['eps'] is not None else "—",
    }

    display_company = resolved_meta.get("name") or data["company"]

    return {
        "company": display_company,
        "exchange": resolved_meta.get("exchange", ""),
        "ticker": data["ticker"],
        "prediction": verdict["label"],
        "predictionColor": verdict["color"],
        "reasons": verdict["reasons"],
        "metrics": disp,
        "raw": {"price": metrics["price"], "intrinsicValue": metrics["intrinsicValue"], "marginPct": margin_pct},
        "priceHistory": price_history
    }
