"""
services/sessions.py — session directory helpers.

All filesystem paths for a session live here.
When switching to a database (Stage 5), only this file changes for
file-listing logic; retrieval/store.py changes for index storage.
"""

from pathlib import Path


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
    """Return True if a FAISS index exists for this session."""
    return Path(f"vector_store/{session_id}/index.faiss").exists()


def delete_index(session_id: str) -> None:
    """Delete all vector store files for a session (called when last file deleted)."""
    store_dir = Path(f"vector_store/{session_id}")
    if store_dir.exists():
        for f in store_dir.iterdir():
            f.unlink()
