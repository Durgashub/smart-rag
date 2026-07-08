"""
services/sessions.py — session directory helpers + index existence check.

Stage 5 update: has_index() now checks PostgreSQL instead of FAISS files.
File storage (docs/) remains on Railway disk — only vectors moved to DB.
"""

from pathlib import Path
from retrieval.store import has_chunks


def get_session_dirs(session_id: str) -> tuple[Path, Path]:
    docs_dir  = Path(f"docs/{session_id}")
    store_dir = Path(f"vector_store/{session_id}")
    docs_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir, store_dir


def get_files_for_session(session_id: str) -> list[dict]:
    docs_dir = Path(f"docs/{session_id}")
    if not docs_dir.exists():
        return []
    return [
        {"name": f.name, "size": f.stat().st_size}
        for f in docs_dir.iterdir()
        if f.is_file()
    ]


def has_index(session_id: str) -> bool:
    """Check PostgreSQL for chunks — replaces FAISS file existence check."""
    return has_chunks(session_id)


def delete_index(session_id: str) -> None:
    """Delete all chunks for this session from PostgreSQL."""
    from ingestion.embeddings import delete_chunks_for_session
    delete_chunks_for_session(session_id)
