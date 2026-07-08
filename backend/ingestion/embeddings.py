"""
ingestion/embeddings.py — batched embedding + PostgreSQL storage.

★ This file replaced FAISS (Stage 4) with PostgreSQL + pgvector (Stage 5).

Previously:
    faiss.write_index(index, f"vector_store/{session_id}/index.faiss")

Now:
    INSERT INTO chunks (session_id, source, text, child_text, parent_id, embedding)
    VALUES (%s, %s, %s, %s, %s, %s)

The embed_texts() function is unchanged — it's just OpenAI API calls.
Only the storage destination changed.
"""

import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from openai import OpenAI
from config import EMBED_MODEL

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client       = OpenAI()
DATABASE_URL = os.getenv("DATABASE_URL")


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


def delete_chunks_for_session(session_id: str) -> int:
    """
    Delete ALL chunks for a session from PostgreSQL.

    Called when a user deletes their last file.
    Returns the number of rows deleted.
    """
    conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=15)
    register_vector(conn)
    cur  = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"  [DB] Deleted {deleted} chunks for session {session_id[:8]}...")
    return deleted


def delete_chunks_for_file(session_id: str, filename: str) -> int:
    """
    Delete chunks for ONE specific file within a session.

    Called when a user deletes a single file but keeps others.
    Returns the number of rows deleted.
    """
    conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=15)
    register_vector(conn)
    cur  = conn.cursor()
    cur.execute(
        "DELETE FROM chunks WHERE session_id = %s AND source = %s",
        (session_id, filename)
    )
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"  [DB] Deleted {deleted} chunks for {filename}")
    return deleted


def save_to_postgres(
    session_id: str,
    embeddings: list[list[float]],
    metadata:   list[dict],
) -> None:
    """
    Save child chunk embeddings + metadata to PostgreSQL.

    Each row in the chunks table:
      session_id  — UUID isolating this user's data
      source      — original filename
      text        — parent chunk (1200 chars, sent to GPT)
      child_text  — child chunk (400 chars, embedded + cross-encoder)
      parent_id   — links child back to its parent group
      embedding   — 1536-dim vector(1536) column

    Clears existing chunks for this session first (full re-index on upload).
    Inserts in batches of 50 for efficiency.
    """
    if not embeddings:
        return

    conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=15)
    register_vector(conn)
    cur  = conn.cursor()

    # Clear existing chunks for this session (fresh re-index)
    cur.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
    deleted = cur.rowcount
    if deleted > 0:
        print(f"  [DB] Cleared {deleted} existing chunks for session")

    # Batch insert
    batch_size = 50
    for i in range(0, len(embeddings), batch_size):
        batch_emb  = embeddings[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]
        rows = [
            (
                session_id,
                m["source"],
                m["text"],
                m["child_text"],
                m["parent_id"],
                emb,
            )
            for m, emb in zip(batch_meta, batch_emb)
        ]
        cur.executemany("""
            INSERT INTO chunks
                (session_id, source, text, child_text, parent_id, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, rows)

    conn.commit()
    cur.close()
    conn.close()
    print(f"  [DB] Saved {len(embeddings)} chunks to PostgreSQL")