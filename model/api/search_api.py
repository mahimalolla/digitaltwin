from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import torch, json
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Netlify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

llm = Llama(
    model_path="model/llms/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=1  # Set to 0 if you're running on CPU
)


# Load documents and embeddings
docs = []
embeddings = []
with open("data/final/embeddings.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        docs.append(obj)
        embeddings.append(obj["embedding"])
corpus_embeddings = torch.tensor(embeddings)

class Query(BaseModel):
    query: str


def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are Warren Buffett's financial digital twin, trained on his shareholder letters, interviews, and writings.

Use the following context to answer the user's question concisely and accurately.

Context:
{context}

Question:
{query}

Answer:"""
    output = llm(prompt.strip(), max_tokens=300, stop=["</s>"])
    return output["choices"][0]["text"].strip()


@app.post("/search")
def search_chunks(query: Query):
    query_embedding = model.encode(query.query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

    top_results = []
    for hit in hits:
        doc = docs[hit["corpus_id"]]
        top_results.append({
            "score": round(hit["score"], 4),
            "source": doc.get("source", ""),
            "text": doc["text"]
        })

    # Combine context from top documents
    context = "\n".join([doc["text"] for doc in top_results])
    answer = generate_answer(query.query, context)

    return {
        "query": query.query,
        "answer": answer,
        "results": top_results
    }
