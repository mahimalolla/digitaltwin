import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

# Paths
EMBEDDINGS_FILE = Path("../data/final/embeddings.jsonl")

# Load model
print("Loading embedding model..")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunks + embeddings
print("Loading saved embeddings..")
docs = []
embeddings = []

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        docs.append(obj)
        embeddings.append(obj["embedding"])

# Convert list of embeddings to tensor
corpus_embeddings = torch.tensor(embeddings)

# Ask query
query = input("Enter your query: ")
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute similarity
hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

# Show results
print("\nTop 5 matching chunks:\n")
for rank, hit in enumerate(hits, 1):
    doc = docs[hit["corpus_id"]]
    print(f"{rank}. (Score: {hit['score']:.4f}) — Source: {doc.get('source', '')}")
    print(doc["text"])
    print("-" * 80)
