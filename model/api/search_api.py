# model/search_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend access (Netlify, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will change this to  frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
docs = []
embeddings = []

emb_path = Path("../data/final/embeddings.jsonl")
if not emb_path.exists():
    raise FileNotFoundError("embeddings.jsonl not found. Run embedding script first.")

with open(emb_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        docs.append(obj)
        embeddings.append(obj["embedding"])

corpus_embeddings = torch.tensor(embeddings)

# Define input schema
class Query(BaseModel):
    query: str

@app.post("/search")
def search_chunks(query: Query):
    query_embedding = model.encode(query.query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

    results = []
    for hit in hits:
        doc = docs[hit["corpus_id"]]
        results.append({
            "score": round(hit["score"], 4),
            "source": doc.get("source", ""),
            "text": doc["text"]
        })

    return {"results": results}
