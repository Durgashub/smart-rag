"""
ingestion/chunking.py — parent-child chunking and low-value chunk filter.

Parent chunks (1200 chars) → sent to GPT for richer answers.
Child chunks  (400 chars)  → embedded for retrieval + cross-encoder scoring.
"""

import re
from config import PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE, CHILD_OVERLAP, CHUNK_OVERLAP


def is_low_value_chunk(text: str) -> bool:
    """
    Return True for chunks that add noise without useful content.

    Filtered out:
    - Bare CONCEPT CHECK question stubs (markdown table rows, no explanations)
    - Citation/footnote blocks (>50% lines are numbered bibliography entries)
    """
    stripped = text.strip()
    if not stripped:
        return True
    lines = [l for l in stripped.splitlines() if l.strip()]
    if not lines:
        return True
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    if table_lines == len(lines) and "CONCEPT CHECK" in stripped.upper():
        return True
    numbered = sum(1 for l in lines if re.match(r"^\d{1,3}[\.\)]\s", l.strip()))
    if len(lines) >= 3 and numbered / len(lines) > 0.5:
        return True
    return False


def chunk_into_parents(text: str) -> list[str]:
    """
    Split text into parent chunks at paragraph boundaries.

    Paragraph-aware (splits on double newlines) rather than fixed char width —
    keeps sentences and tables intact far more often.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > PARENT_CHUNK_SIZE:
            flush()
            start = 0
            while start < len(para):
                chunks.append(para[start:start + PARENT_CHUNK_SIZE].strip())
                start += PARENT_CHUNK_SIZE - CHUNK_OVERLAP
            continue
        if len(current) + len(para) + 2 <= PARENT_CHUNK_SIZE:
            current = f"{current}\n\n{para}" if current else para
        else:
            flush()
            current = para
    flush()
    return [c for c in chunks if c]


def split_into_children(parent: str) -> list[str]:
    """
    Split a parent chunk into smaller child chunks for embedding.

    Children are used for:
    - FAISS vector search (short text → more precise embeddings)
    - Cross-encoder re-ranking (model trained on ~400-char passages)
    """
    children = []
    start    = 0
    while start < len(parent):
        child = parent[start:start + CHILD_CHUNK_SIZE].strip()
        if len(child) > 50:  # skip tiny fragments
            children.append(child)
        start += CHILD_CHUNK_SIZE - CHILD_OVERLAP
    return children


def build_parent_child_chunks(text: str) -> list[dict]:
    """
    Build parent-child chunk pairs from document text.

    Returns list of dicts:
    {
        "child_text":  "...400 char chunk for retrieval...",
        "parent_text": "...1200 char chunk sent to GPT...",
        "parent_id":   0,
    }
    """
    parents = chunk_into_parents(text)
    result  = []
    for parent_id, parent in enumerate(parents):
        if is_low_value_chunk(parent):
            continue
        for child in split_into_children(parent):
            if not is_low_value_chunk(child):
                result.append({
                    "child_text":  child,
                    "parent_text": parent,
                    "parent_id":   parent_id,
                })
    return result
