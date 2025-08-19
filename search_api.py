# search_api.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from pathlib import Path
import torch, json

# --- NEW: fundamentals ---
import yfinance as yf
import numpy as np
import pandas as pd

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- RAG pieces (existing) -----
model = SentenceTransformer("all-MiniLM-L6-v2")

llm = Llama(
    model_path="model/llms/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=0  # set >0 if you have VRAM; 0 is safest
)

docs, embeddings = [], []
emb_path = Path("data/final/embeddings.jsonl")
if emb_path.exists():
    with emb_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(obj)
            embeddings.append(obj["embedding"])
    corpus_embeddings = torch.tensor(embeddings)
else:
    # empty corpus fallback
    corpus_embeddings = torch.zeros((1, 384))
    docs = [{"text": "No corpus loaded yet.", "source": ""}]

class QueryIn(BaseModel):
    query: str

def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are Warren Buffett's financial digital twin.

Use the following context to answer concisely and cite key ideas:

Context:
{context}

Question:
{query}

Answer:"""
    out = llm(prompt.strip(), max_tokens=300, stop=["</s>"])
    return out["choices"][0]["text"].strip()

@app.post("/search")
def search_chunks(query: QueryIn):
    q_emb = model.encode(query.query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, corpus_embeddings, top_k=5)[0]
    top_results = []
    for hit in hits:
        idx = int(hit["corpus_id"])
        doc = docs[idx] if 0 <= idx < len(docs) else {"text": "", "source": ""}
        top_results.append({
            "score": round(float(hit["score"]), 4),
            "source": doc.get("source", ""),
            "text": doc.get("text", "")
        })
    context = "\n".join(d["text"] for d in top_results if d["text"])
    answer = generate_answer(query.query, context)
    return {"query": query.query, "answer": answer, "results": top_results}

# ----- Predictor pieces (NEW) -----

def _safe_get(series_or_df, key):
    try:
        return float(series_or_df.loc[key].iloc[0])
    except Exception:
        return None

def compute_metrics(ticker: str):
    tk = yf.Ticker(ticker)
    # Basic identity
    info = {}
    try:
        info = tk.get_info()  # works on recent yfinance
    except Exception:
        try:
            info = tk.info  # older fallback
        except Exception:
            info = {}
    company = info.get("longName") or info.get("shortName") or ticker.upper()

    # Financial statements (annual)
    try:
        inc = tk.financials  # Income Statement
        bal = tk.balance_sheet
    except Exception:
        inc = pd.DataFrame()
        bal = pd.DataFrame()

    net_income = _safe_get(inc, "Net Income")
    equity = _safe_get(bal, "Total Stockholder Equity") or _safe_get(bal, "Total Equity Gross Minority Interest")
    total_liab = _safe_get(bal, "Total Liab")

    # Shares & price
    fast = getattr(tk, "fast_info", {}) or {}
    shares = None
    for key in ("sharesOutstanding", "shares_outstanding"):
        if key in fast and fast[key]:
            shares = float(fast[key])
            break
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

    # EPS (TTM approx)
    eps = None
    try:
        eps = float(info.get("trailingEps")) if info.get("trailingEps") is not None else None
    except Exception:
        eps = None
    if eps is None and net_income and shares:
        eps = net_income / shares

    # P/E
    pe = None
    for key in ("peRatio", "trailingPE"):
        try:
            val = fast.get(key) if key in fast else info.get(key)
            if val: pe = float(val); break
        except Exception:
            pass
    if pe is None and price and eps and eps != 0:
        pe = price / eps

    # ROE
    roe = None
    if net_income is not None and equity not in (None, 0):
        roe = net_income / equity

    # Debt/Equity
    d_to_e = None
    if total_liab is not None and equity not in (None, 0):
        d_to_e = total_liab / equity

    # Earnings growth (CAGR from net income if we have multiple years)
    growth = None
    try:
        if not inc.empty and "Net Income" in inc.index and inc.shape[1] >= 2:
            ni = inc.loc["Net Income"].dropna().astype(float)
            ni = ni.sort_index()  # oldest -> newest
            if len(ni) >= 2:
                n_years = max(1, len(ni)-1)
                start, end = float(ni.iloc[0]), float(ni.iloc[-1])
                if start > 0 and end > 0:
                    growth = (end / start) ** (1 / n_years) - 1
    except Exception:
        pass

    # Intrinsic value (very rough Graham-style)
    intrinsic = None
    if eps is not None:
        g = (growth or 0.08) * 100  # convert to % assumption if missing
        Y = 4.4  # base yield constant
        intrinsic = eps * (8.5 + 2 * g) * 4.4 / Y

    return {
        "company": company,
        "ticker": ticker.upper(),
        "metrics": {
            "roe": roe,                 # fraction (e.g., 0.22)
            "pe": pe,                   # ratio
            "debtToEquity": d_to_e,     # ratio
            "earningsGrowth": growth,   # fraction
            "eps": eps,
            "price": price,
            "intrinsicValue": intrinsic
        }
    }

def score_buffett(metrics: dict):
    """Simple Buffett-ish rules -> label + reasons."""
    roe = metrics["roe"]
    pe = metrics["pe"]
    dte = metrics["debtToEquity"]
    g = metrics["earningsGrowth"]
    price = metrics["price"]
    iv = metrics["intrinsicValue"]

    reasons = []
    score = 0

    if roe is not None:
        if roe >= 0.15: score += 2; reasons.append(f"ROE {roe:.1%} ≥ 15% (strong)")
        elif roe >= 0.10: score += 1; reasons.append(f"ROE {roe:.1%} ≥ 10% (ok)")
        else: reasons.append(f"ROE {roe:.1%} < 10% (weak)")

    if pe is not None:
        if pe <= 20: score += 2; reasons.append(f"P/E {pe:.1f} ≤ 20 (reasonable)")
        elif pe <= 25: score += 1; reasons.append(f"P/E {pe:.1f} ≤ 25 (borderline)")
        else: reasons.append(f"P/E {pe:.1f} > 25 (expensive)")

    if dte is not None:
        if dte <= 0.5: score += 2; reasons.append(f"Debt/Equity {dte:.2f} ≤ 0.5 (conservative)")
        elif dte <= 1.0: score += 1; reasons.append(f"Debt/Equity {dte:.2f} ≤ 1.0 (moderate)")
        else: reasons.append(f"Debt/Equity {dte:.2f} > 1.0 (high)")

    if g is not None:
        if g >= 0.10: score += 2; reasons.append(f"Earnings growth {g:.1%} ≥ 10% (healthy)")
        elif g >= 0.05: score += 1; reasons.append(f"Earnings growth {g:.1%} ≥ 5% (ok)")
        else: reasons.append(f"Earnings growth {g:.1%} < 5% (low)")

    # valuation vs intrinsic
    if iv is not None and price is not None:
        margin = (iv - price) / price
        if margin >= 0.25: score += 2; reasons.append(f"Price below intrinsic by {margin:.0%} (value)")
        elif margin >= 0.0: score += 1; reasons.append(f"Near intrinsic (fair)")
        else: reasons.append(f"Above intrinsic by {abs(margin):.0%} (premium)")

    # Map score -> label
    if score >= 8: label, color = "Likely Buy", "green"
    elif score >= 5: label, color = "Hold / Watchlist", "yellow"
    else: label, color = "Avoid / Too Expensive", "red"

    return {"score": score, "label": label, "color": color, "reasons": reasons}

@app.get("/predict")
def predict(ticker: str = Query(..., min_length=1)):
    try:
        data = compute_metrics(ticker.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch data for {ticker}: {e}")

    metrics = data["metrics"]
    if all(v is None for v in metrics.values()):
        raise HTTPException(status_code=404, detail=f"No fundamentals available for {ticker.upper()}")

    verdict = score_buffett(metrics)

    # Add price history (last ~6 months of daily closes)
    price_history = []
    try:
        tk = yf.Ticker(ticker.strip())
        hist = tk.history(period="6mo", interval="1d")
        if not hist.empty and "Close" in hist:
            price_history = [float(x) for x in hist["Close"].fillna(method="ffill").tail(180).tolist()]
    except Exception:
        price_history = []

    # Valuation margin vs intrinsic
    margin_pct = None
    if metrics["intrinsicValue"] is not None and metrics["price"] is not None and metrics["price"] != 0:
        margin_pct = (metrics["intrinsicValue"] - metrics["price"]) / metrics["price"]  # + = undervalued

    # Pretty strings for UI
    disp = {
        "roe": f"{metrics['roe']*100:.1f}%" if metrics['roe'] is not None else "—",
        "pe": f"{metrics['pe']:.1f}" if metrics['pe'] is not None else "—",
        "debtToEquity": f"{metrics['debtToEquity']:.2f}" if metrics['debtToEquity'] is not None else "—",
        "earningsGrowth": f"{metrics['earningsGrowth']*100:.1f}%" if metrics['earningsGrowth'] is not None else "—",
        "intrinsicValue": f"${metrics['intrinsicValue']:.2f}" if metrics['intrinsicValue'] is not None else "—",
        "price": f"${metrics['price']:.2f}" if metrics['price'] is not None else "—",
        "eps": f"{metrics['eps']:.2f}" if metrics['eps'] is not None else "—",
    }

    return {
        "company": data["company"],
        "ticker": data["ticker"],
        "prediction": verdict["label"],
        "predictionColor": verdict["color"],
        "reasons": verdict["reasons"],
        "metrics": disp,
        "raw": {  # keep raw numerics for calculations if you want later
            "price": metrics["price"],
            "intrinsicValue": metrics["intrinsicValue"],
            "marginPct": margin_pct
        },
        "priceHistory": price_history
    }
