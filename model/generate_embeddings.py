from sentence_transformers import SentenceTransformer
import json
from pathlib import Path
from tqdm import tqdm
import os

# Paths
CHUNKS_PATH = Path("../data/final/chunks.jsonl")
EMBEDDINGS_PATH = Path("../data/final/embeddings.jsonl")
EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load model
print("Loading local embedding model..")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read input chunks
print("Reading chunks..")
chunks = []
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Generate and save embeddings
print("Generating embeddings..")
with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f_out:
    for chunk in tqdm(chunks):
        text = chunk["text"]
        embedding = model.encode(text).tolist()
        chunk["embedding"] = embedding
        f_out.write(json.dumps(chunk) + "\n")

print(f"Done! Saved {len(chunks)} embeddings to: {EMBEDDINGS_PATH}")
