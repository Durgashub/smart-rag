"""
services/map_reduce.py — cross-document map-reduce answering.

  simple_retrieve()    — raw FAISS search bypassing Stage 3 (avoids HyDE hallucination)
  get_doc_chunks()     — 3-level per-document retrieval strategy
  map_reduce_answer()  — MAP: answer per doc in isolation, REDUCE: merge answers
"""

import json
import re
from pathlib import Path

from config import client, CHAT_MODEL
from services.accuracy import avg_accuracy, calculate_accuracy
from services.generation import build_messages_with_history
from retrieval.store import embed_query, search_chunks


# ── Simple retrieval (bypasses Stage 3 pipeline) ─────────────────────────────

def simple_retrieve(question: str, session_id: str, top_k: int = 20) -> list[dict]:
    """
    Raw pgvector search — no rewriting, no HyDE, no MMR, no cross-encoder.

    Used by the map phase to avoid HyDE hallucinating names like "Alex Johnson"
    for "who are the candidates?", which would poison per-document retrieval.
    """
    try:
        query_vec = embed_query(question)
        return search_chunks(session_id, query_vec, top_k=top_k)
    except Exception as e:
        print(f"    [SimpleRetrieve] Failed: {e}")
        return []


# ── Per-document chunk retrieval ─────────────────────────────────────────────

def get_doc_chunks(session_id: str, filename: str, question: str, top_k: int) -> list[dict]:
    """
    Retrieve chunks from ONE specific document — 3-level fallback:

    L1 Question-based: simple FAISS search filtered by filename
    L2 Name-based:     search with person's name from filename
    L3 Metadata scan:  read metadata.json directly (always works)

    Uses simple_retrieve() (not retrieve()) so Stage 3 pipeline doesn't run
    and HyDE can't hallucinate a wrong name that redirects search away from
    the correct document.
    """
    # L1
    pool   = simple_retrieve(question, session_id, top_k=top_k * 6)
    chunks = [c for c in pool if c.get("source") == filename]
    if chunks:
        print(f"    [Map] {filename} → {len(chunks)} chunks (L1)")
        return chunks[:top_k]

    # L2
    name_query = re.sub(r"[_\.\-]", " ", filename.rsplit(".", 1)[0]).strip()
    pool2      = simple_retrieve(name_query, session_id, top_k=top_k * 4)
    chunks2    = [c for c in pool2 if c.get("source") == filename]
    if chunks2:
        print(f"    [Map] {filename} → {len(chunks2)} chunks (L2)")
        return chunks2[:top_k]

    # L3
    try:
        with open(f"vector_store/{session_id}/metadata.json", "r", encoding="utf-8") as mf:
            metadata = json.load(mf)
        chunks3 = [m for m in metadata if m.get("source") == filename][:top_k]
        if chunks3:
            print(f"    [Map] {filename} → {len(chunks3)} chunks (L3)")
            return chunks3
    except Exception as e:
        print(f"    [Map] {filename} → L3 failed: {e}")

    print(f"    [Map] {filename} → 0 chunks (all levels failed)")
    return []


# ── Map-reduce ────────────────────────────────────────────────────────────────

def map_reduce_answer(
    question: str,
    session_id: str,
    history: list[dict],
    top_k: int = 8,
) -> dict:
    """
    MAP:    One GPT call per document in complete isolation.
    REDUCE: Merge per-document answers into one final response.

    Cross-contamination is structurally impossible — each map GPT call
    sees only one document's content, so Durga's Selenium test suites
    cannot appear under Sneha's name.
    """
    docs_dir = Path(f"docs/{session_id}")
    if not docs_dir.exists():
        return {"answer": "No documents found.", "sources": [], "chunks": [], "mode": "cross_doc", "accuracy": 0}

    uploaded_files = [f.name for f in sorted(docs_dir.iterdir()) if f.is_file()]
    if not uploaded_files:
        return {"answer": "No documents uploaded.", "sources": [], "chunks": [], "mode": "cross_doc", "accuracy": 0}

    print(f"\n  [MapReduce] {len(uploaded_files)} doc(s) | '{question[:60]}'")

    # ── MAP ───────────────────────────────────────────────────────────────────
    per_doc_answers: list[dict] = []
    all_chunks:      list[dict] = []

    for filename in uploaded_files:
        doc_chunks = get_doc_chunks(session_id, filename, question, top_k)
        if not doc_chunks:
            per_doc_answers.append({"filename": filename, "answer": f"[{filename}]: Could not retrieve content."})
            continue

        all_chunks.extend(doc_chunks)
        context    = "\n\n".join(c["text"] for c in doc_chunks)
        map_system = (
            f"You are reading ONE document: {filename}\n\n"
            "RULES:\n"
            "1. Use ONLY the content provided — nothing else.\n"
            "2. The document belongs to one person. Find their name at the very top.\n"
            "3. Answer the question as fully as possible from this document.\n"
            "4. If information is missing, say what IS present instead of 'Not found'.\n"
            "5. Be concise — 3-8 sentences maximum."
        )
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": map_system},
                    {"role": "user",   "content": f"Document content:\n{context}\n\nQuestion: {question}"},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            doc_answer = resp.choices[0].message.content.strip()
            print(f"    [Map] '{filename}' → {len(doc_answer)} chars")
        except Exception as e:
            doc_answer = f"Error processing {filename}: {e}"

        per_doc_answers.append({"filename": filename, "answer": doc_answer})

    # ── REDUCE ────────────────────────────────────────────────────────────────
    if len(per_doc_answers) == 1:
        final_answer = per_doc_answers[0]["answer"]
    else:
        reduce_context = "\n\n".join(
            f"=== {d['filename']} ===\n{d['answer']}" for d in per_doc_answers
        )
        reduce_system = (
            "You are merging per-document answers into one final response.\n\n"
            "RULES:\n"
            "1. Each document's answer is under its === filename === header.\n"
            "2. Combine into a clear numbered list — one entry per document.\n"
            "3. Do NOT add information not in the per-document answers.\n"
            "4. Do NOT skip any document — list all of them.\n"
            "5. Keep the final answer clear and well-structured."
        )
        reduce_messages = build_messages_with_history(
            system_prompt=reduce_system,
            context=reduce_context,
            question=f"Merge these per-document answers for: {question}",
            history=history,
        )
        try:
            resp         = client.chat.completions.create(
                model=CHAT_MODEL, messages=reduce_messages, temperature=0.4, max_tokens=1500
            )
            final_answer = resp.choices[0].message.content.strip()
            print(f"    [Reduce] {len(final_answer)} chars")
        except Exception as e:
            final_answer = "\n\n".join(f"**{d['filename']}:**\n{d['answer']}" for d in per_doc_answers)
            print(f"    [Reduce] Failed ({e}), concatenation fallback")

    # Deduplicate chunks for display
    seen, unique = set(), []
    for c in all_chunks:
        key = (c.get("source"), c.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return {
        "answer":   final_answer,
        "mode":     "cross_doc",
        "sources":  uploaded_files,
        "accuracy": avg_accuracy(unique) if unique else 55,
        "chunks": [
            {
                "source":   c.get("source", "Unknown"),
                "text":     c["text"][:300],
                "distance": c.get("distance"),
                "accuracy": calculate_accuracy(c.get("distance", 1.0), c.get("cross_encoder_score")),
            }
            for c in unique[:8]
        ],
    }
