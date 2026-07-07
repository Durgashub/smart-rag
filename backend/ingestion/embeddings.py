"""
ingestion/embeddings.py — batched OpenAI embedding calls.

★ TO MIGRATE TO pgvector (Stage 5): this file + retrieval/store.py change.
  embed_texts() stays the same (it's just OpenAI).
  The save logic at the bottom would INSERT INTO chunks instead of writing FAISS.
"""

import json
import numpy as np
import faiss
from config import client, EMBED_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using OpenAI's embedding model.

    Batches in groups of 100 to stay within API rate limits.
    Returns list of 1536-dimensional float vectors.
    """
    if not texts:
        return []
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch    = texts[i:i + 100]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


def save_to_faiss(session_id: str, embeddings: list[list[float]], metadata: list[dict]) -> None:
    """
    Save embeddings + metadata to a FAISS index on disk.

    ★ pgvector migration: replace this function with INSERT INTO chunks SQL.
    """
    import os
    store_dir = f"vector_store/{session_id}"
    os.makedirs(store_dir, exist_ok=True)

    matrix    = np.array(embeddings, dtype="float32")
    index     = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, f"{store_dir}/index.faiss")
    with open(f"{store_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
