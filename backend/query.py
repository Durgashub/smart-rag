"""
Retrieval helpers — session-scoped.

Each user session has its own FAISS index under vector_store/<session_id>/
so queries never cross between users.
"""

import json

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 6


def load_index(session_id: str):
    store_dir = f"vector_store/{session_id}"
    index = faiss.read_index(f"{store_dir}/index.faiss")
    with open(f"{store_dir}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_query(question: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array([response.data[0].embedding], dtype="float32")


def _diversify(candidates: list[dict], top_k: int) -> list[dict]:
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


def retrieve(question: str, session_id: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    index, metadata = load_index(session_id)
    query_vector = embed_query(question)

    candidate_k = min(top_k * 4, len(metadata))
    distances, indices = index.search(query_vector, candidate_k)

    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        candidates.append({**metadata[idx], "distance": float(dist)})

    return _diversify(candidates, top_k)