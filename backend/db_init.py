"""
db_init.py — one-time database setup script.

Run this ONCE to create the chunks table and pgvector index:
    python db_init.py

Safe to run multiple times — uses CREATE IF NOT EXISTS.
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=__file__.replace("db_init.py", ".env"))

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")


def init_db():
    conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=15)
    cur  = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create chunks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id          SERIAL PRIMARY KEY,
            session_id  TEXT        NOT NULL,
            source      TEXT        NOT NULL,
            text        TEXT        NOT NULL,
            child_text  TEXT        NOT NULL,
            parent_id   INTEGER     NOT NULL,
            embedding   vector(1536),
            created_at  TIMESTAMP   DEFAULT NOW()
        );
    """)

    # Index for fast vector search per session
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks
        USING ivfflat (embedding vector_l2_ops)
        WITH (lists = 100);
    """)

    # Index for fast session filtering
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_session_idx
        ON chunks (session_id);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized — chunks table and indexes created.")


if __name__ == "__main__":
    init_db()