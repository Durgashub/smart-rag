"""
ingestion/pipeline.py — ingest_session() orchestrating full document ingestion.

Reads from:  docs/<session_id>/
Writes to:   vector_store/<session_id>/

Called by ingest.py (the CLI wrapper that server.py subprocess-calls on upload).
"""

import glob
from pathlib import Path

from ingestion.loaders  import load_text_from_file
from ingestion.chunking import build_parent_child_chunks, chunk_into_parents, is_low_value_chunk
from ingestion.embeddings import embed_texts, save_to_faiss
from config import SUPPORTED_EXTENSIONS


def ingest_session(session_id: str) -> None:
    docs_dir  = f"docs/{session_id}"
    store_dir = f"vector_store/{session_id}"

    all_files  = glob.glob(f"{docs_dir}/**/*.*", recursive=True)
    file_paths = [p for p in all_files if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]

    if not file_paths:
        print(f"No supported files found in '{docs_dir}/'.")
        return

    all_child_texts: list[str]  = []
    metadata:        list[dict] = []
    total_parents  = 0
    total_children = 0

    for path in file_paths:
        ext  = Path(path).suffix.lower()
        print(f"Loading {path} ({ext}) ...")
        text = load_text_from_file(path)

        if not text.strip():
            print("  -> No text extracted, skipping.")
            continue

        pairs = build_parent_child_chunks(text)
        if not pairs:
            print("  -> No chunks produced, skipping.")
            continue

        unique_parents = len(set(p["parent_id"] for p in pairs))
        skipped_raw    = len(chunk_into_parents(text)) - unique_parents
        total_parents  += unique_parents
        total_children += len(pairs)

        print(
            f"  -> {unique_parents} parent chunks → "
            f"{len(pairs)} child chunks "
            f"({skipped_raw} low-value parents skipped)"
        )

        for pair in pairs:
            all_child_texts.append(pair["child_text"])
            metadata.append({
                "source":     Path(path).name,
                "text":       pair["parent_text"],   # GPT gets the full parent
                "child_text": pair["child_text"],     # cross-encoder + debug
                "parent_id":  pair["parent_id"],
            })

    if not all_child_texts:
        print("No child chunks produced. Check file contents.")
        return

    print(f"\nTotal: {total_parents} parents → {total_children} children across {len(file_paths)} file(s)")
    print(f"Embedding {len(all_child_texts)} child chunks (used for retrieval)...")

    embeddings = embed_texts(all_child_texts)
    save_to_faiss(session_id, embeddings, metadata)

    print(f"Done. Index saved to '{store_dir}/'.")
    print(f"  Child chunks indexed: {len(all_child_texts)}")
    print(f"  GPT will receive parent chunks for richer answers.")
