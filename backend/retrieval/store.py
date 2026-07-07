"""
retrieval/store.py — FAISS index load + query embedding.

★ TO MIGRATE TO pgvector (Stage 5): only THIS file changes.
  Replace load_index() with a psycopg2 connection + query.
  Replace embed_query() stays the same (it's just OpenAI).
  Everything else in the retrieval pipeline is vector-DB agnostic.
"""

import json
import numpy as np
import faiss
from config import client, EMBED_MODEL


def load_index(session_id: str) -> tuple:
    """
    Load FAISS index and metadata for a session.

    Returns (faiss.Index, list[dict]).
    Each metadata dict: {source, text (parent), child_text, parent_id}
    """
    store_dir = f"vector_store/{session_id}"
    index = faiss.read_index(f"{store_dir}/index.faiss")
    with open(f"{store_dir}/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_query(question: str) -> np.ndarray:
    """Embed a single query string → (1, 1536) float32 array for FAISS."""
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array([response.data[0].embedding], dtype="float32")
