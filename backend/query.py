"""
Retrieval helpers: embed a question and find the most relevant chunks
from the FAISS index built by ingest.py.

Retrieval is source-diversified: when a question could be answered from
more than one document, this avoids letting one document's chunks crowd
out every slot just because it happens to have more total content indexed.
"""

import json

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

STORE_DIR = "vector_store"
EMBED_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 6


def load_index():
    index = faiss.read_index(f"{STORE_DIR}/index.faiss")
    with open(f"{STORE_DIR}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array([response.data[0].embedding], dtype="float32")


def _diversify(candidates: list[dict], top_k: int) -> list[dict]:
    """Greedily select up to top_k candidates (already sorted by
    relevance), capping how many can come from any single source so a
    question needing info from multiple documents doesn't end up starved
    because one document's chunks dominate the nearest neighbors."""
    selected = []
    per_source_count = {}
    max_per_source = max(2, top_k // 2)

    for c in candidates:
        if len(selected) >= top_k:
            break
        src = c["source"]
        if per_source_count.get(src, 0) >= max_per_source:
            continue
        selected.append(c)
        per_source_count[src] = per_source_count.get(src, 0) + 1

    if len(selected) < top_k:
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[: top_k - len(selected)])

    return selected


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    index, metadata = load_index()
    query_vector = embed_query(question)

    # Pull a wider candidate pool than we'll actually use, so the
    # diversity step has room to swap in a relevant-but-not-top-4 chunk
    # from a different document.
    candidate_k = min(top_k * 4, len(metadata))
    distances, indices = index.search(query_vector, candidate_k)

    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        candidates.append({**metadata[idx], "distance": float(dist)})

    return _diversify(candidates, top_k)
