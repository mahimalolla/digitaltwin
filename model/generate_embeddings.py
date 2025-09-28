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

