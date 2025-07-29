import json
from pathlib import Path
from typing import List, Dict
import sys
from pathlib import Path
import re

def clean_text(text: str) -> str:
    """Remove extra whitespace and control characters."""
    text = re.sub(r"\s+", " ", text)  # Replace multiple whitespace with single space
    return text.strip()
# Paths
STRUCTURED_DIR = Path("../data/structured")
FLATTENED_PATH = Path("../data/final/chunks.jsonl")
FLATTENED_PATH.parent.mkdir(parents=True, exist_ok=True)

def flatten_chunks() -> List[Dict]:
    all_chunks = []
    
    for file in STRUCTURED_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            doc = json.load(f)
            doc_id = doc.get("id", file.stem)
            year = doc.get("year")
            source = doc.get("source")
            type_ = doc.get("type", "unknown")
            chunks = doc.get("chunks", [])

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc_id}-{i}",
                    "text": clean_text(chunk),
                    "source": source,
                    "year": year,
                    "type": type_
                })

    return all_chunks

def save_chunks_jsonl(chunks: List[Dict]):
    with open(FLATTENED_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    chunks = flatten_chunks()
    print(f"Flattened {len(chunks)} chunks.")
    save_chunks_jsonl(chunks)
    print(f"Saved to: {FLATTENED_PATH}")
