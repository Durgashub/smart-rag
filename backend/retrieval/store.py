"""
retrieval/store.py — pgvector vector search.

★ This file replaced FAISS (Stage 4) with PostgreSQL + pgvector (Stage 5).

Previously:
    index = faiss.read_index(f"vector_store/{session_id}/index.faiss")
    distances, indices = index.search(query_vec, k)

Now:
    SELECT source, text, child_text, parent_id,
           embedding <-> %s AS distance
    FROM chunks
    WHERE session_id = %s
    ORDER BY distance
    LIMIT %s

Benefits:
  - Vectors persist across Railway restarts (no more data loss on redeploy)
  - Filter by session_id natively in SQL
  - IVFFlat index keeps search fast as corpus grows
  - Single PostgreSQL instance serves all sessions
"""

import os
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from openai import OpenAI
from config import EMBED_MODEL

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client       = OpenAI()
DATABASE_URL = os.getenv("DATABASE_URL")


def _get_conn():
    """Get a database connection with pgvector registered."""
    conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=15)
    register_vector(conn)
    return conn


def embed_query(question: str) -> np.ndarray:
    """Embed a single query string → numpy array for pgvector."""
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return np.array(response.data[0].embedding, dtype="float32")


def load_index(session_id: str) -> tuple:
    """
    Load all chunks for a session from PostgreSQL.

    Returns (None, metadata) — same interface as the FAISS version.
    The None replaces the faiss.Index object; pgvector handles search via
    search_chunks() which is called directly by retrieval/pipeline.py.

    metadata list format: [{source, text, child_text, parent_id}, ...]
    """
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT source, text, child_text, parent_id
        FROM chunks
        WHERE session_id = %s
        ORDER BY id
    """, (session_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    metadata = [
        {
            "source":     row[0],
            "text":       row[1],
            "child_text": row[2],
            "parent_id":  row[3],
        }
        for row in rows
    ]
    return None, metadata


def search_chunks(
    session_id: str,
    query_embedding: np.ndarray,
    top_k: int = 20,
) -> list[dict]:
    """
    Vector similarity search using pgvector L2 distance operator (<->).

    Returns chunks sorted by distance (closest first).
    Each dict: {source, text, child_text, parent_id, distance}
    """
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT source, text, child_text, parent_id,
               (embedding <-> %s) AS distance
        FROM chunks
        WHERE session_id = %s
        ORDER BY distance
        LIMIT %s
    """, (query_embedding.tolist(), session_id, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "source":     row[0],
            "text":       row[1],
            "child_text": row[2],
            "parent_id":  row[3],
            "distance":   float(row[4]),
        }
        for row in rows
    ]


def has_chunks(session_id: str) -> bool:
    """Return True if this session has any indexed chunks in the DB."""
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT 1 FROM chunks WHERE session_id = %s LIMIT 1", (session_id,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists