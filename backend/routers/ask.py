"""
routers/ask.py — main chat endpoints: /api/ask and /api/ask/stream (SSE).

Pipeline for each request:
  1. classify_intent()         → services/intent.py
  2. cross_doc → map_reduce()  → services/map_reduce.py
  3. out_of_scope → refuse
  4. Otherwise → retrieve()    → retrieval/pipeline.py
  5. build_messages_with_history() → services/generation.py
  6. GPT completion
  7. verify_answer()           → services/generation.py
"""

import json
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse

from config import client, CHAT_MODEL
from schemas import QuestionRequest
from prompts import get_system_prompt, build_context_prompt
from services.sessions import has_index
from services.intent import classify_intent
from services.generation import build_messages_with_history, verify_answer, get_top_k
from services.accuracy import calculate_accuracy, avg_accuracy
from services.map_reduce import map_reduce_answer, get_doc_chunks
from retrieval.pipeline import retrieve

router = APIRouter()


def _collect_sources(chunks: list[dict]) -> list[str]:
    seen, sources = set(), []
    for c in chunks:
        src = c.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    return sources


def _format_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source":   c.get("source", "Unknown"),
            "text":     c["text"][:300],
            "distance": c.get("distance"),
            "accuracy": calculate_accuracy(c.get("distance", 1.0), c.get("cross_encoder_score")),
        }
        for c in chunks
    ]


@router.post("/api/ask")
def ask_question(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None),
):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    if not has_index(x_session_id):
        return {"answer": "No documents indexed yet. Please upload a document first.",
                "sources": [], "chunks": [], "mode": "none", "accuracy": 0}

    history = [{"role": t.role, "content": t.content} for t in request.history]
    intent  = classify_intent(request.question, history)

    if request.mode and request.mode != "auto":
        intent.type = request.mode
        print(f"  [Intent] Overridden by frontend: {intent.type}")

    if intent.type == "out_of_scope":
        return {"answer": "I can only answer questions about your uploaded documents.",
                "sources": [], "chunks": [], "mode": "out_of_scope", "accuracy": 0}

    if intent.type == "cross_doc":
        return map_reduce_answer(request.question, x_session_id, history,
                                 top_k=get_top_k(intent.type, request.question))

    top_k  = get_top_k(intent.type, request.question)
    chunks = retrieve(request.question, session_id=x_session_id, top_k=top_k)
    if not chunks:
        return {"answer": "No relevant content found in your documents.",
                "sources": [], "chunks": [], "mode": intent.type, "accuracy": 0}

    messages = build_messages_with_history(
        system_prompt=get_system_prompt(intent.type),
        context=build_context_prompt(request.question, chunks, intent.type),
        question=request.question,
        history=history,
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL, messages=messages,
        temperature=0.3 if intent.type in ("analyzer", "skill_gap") else 0.7,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content.strip()

    if intent.type in ("single_doc", "identity", "general"):
        answer, _ = verify_answer(request.question, answer, chunks)

    return {
        "answer":      answer,
        "mode":        intent.type,
        "is_followup": intent.is_followup,
        "sources":     _collect_sources(chunks),
        "accuracy":    avg_accuracy(chunks),
        "chunks":      _format_chunks(chunks),
    }


@router.post("/api/ask/stream")
async def ask_question_stream(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None),
):
    """
    SSE streaming version of /api/ask.
    Token events:  data: {"type": "token", "content": "word "}
    Done event:    data: {"type": "done", "mode": "...", "sources": [...], "accuracy": 84}
    """
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")

    if not has_index(x_session_id):
        async def no_docs():
            yield "data: " + json.dumps({"type": "token", "content": "No documents indexed yet."}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "mode": "none", "sources": [], "accuracy": 0, "is_followup": False}) + "\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    history = [{"role": t.role, "content": t.content} for t in request.history]

    async def generate():
        intent = classify_intent(request.question, history)
        if request.mode and request.mode != "auto":
            intent.type = request.mode

        # out_of_scope
        if intent.type == "out_of_scope":
            yield "data: " + json.dumps({"type": "token", "content": "I can only answer questions about your uploaded documents."}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "mode": "out_of_scope", "sources": [], "accuracy": 0, "is_followup": False}) + "\n\n"
            return

        # cross_doc: map phase (blocking) → stream reduce phase
        if intent.type == "cross_doc":
            docs_dir       = Path(f"docs/{x_session_id}")
            uploaded_files = [f.name for f in sorted(docs_dir.iterdir()) if f.is_file()] if docs_dir.exists() else []
            per_doc, all_chunks = [], []
            top_k_map = get_top_k(intent.type, request.question)

            for filename in uploaded_files:
                doc_chunks = get_doc_chunks(x_session_id, filename, request.question, top_k_map)
                if not doc_chunks:
                    per_doc.append({"filename": filename, "answer": f"[{filename}]: No content."})
                    continue
                all_chunks.extend(doc_chunks)
                ctx = "\n\n".join(c["text"] for c in doc_chunks)
                try:
                    r = client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[
                            {"role": "system", "content": f"Answer ONLY from: {filename}. Be concise."},
                            {"role": "user",   "content": f"{ctx}\n\nQuestion: {request.question}"},
                        ],
                        temperature=0.3, max_tokens=600,
                    )
                    per_doc.append({"filename": filename, "answer": r.choices[0].message.content.strip()})
                except Exception as e:
                    per_doc.append({"filename": filename, "answer": f"Error: {e}"})

            reduce_ctx  = "\n\n".join(f"=== {d['filename']} ===\n{d['answer']}" for d in per_doc)
            reduce_msgs = build_messages_with_history(
                system_prompt="Merge per-document answers. List one entry per document. Do not skip any.",
                context=reduce_ctx,
                question=f"Merge for: {request.question}",
                history=history,
            )
            stream = client.chat.completions.create(
                model=CHAT_MODEL, messages=reduce_msgs, temperature=0.4, max_tokens=1500, stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield "data: " + json.dumps({"type": "token", "content": token}) + "\n\n"

            yield "data: " + json.dumps({
                "type": "done", "mode": "cross_doc", "sources": uploaded_files,
                "accuracy": avg_accuracy(all_chunks) if all_chunks else 55,
                "is_followup": intent.is_followup,
            }) + "\n\n"
            return

        # standard retrieval + streaming GPT
        top_k  = get_top_k(intent.type, request.question)
        chunks = retrieve(request.question, session_id=x_session_id, top_k=top_k)
        if not chunks:
            yield "data: " + json.dumps({"type": "token", "content": "No relevant content found."}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "mode": intent.type, "sources": [], "accuracy": 0, "is_followup": False}) + "\n\n"
            return

        messages = build_messages_with_history(
            system_prompt=get_system_prompt(intent.type),
            context=build_context_prompt(request.question, chunks, intent.type),
            question=request.question,
            history=history,
        )
        stream = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages,
            temperature=0.3 if intent.type in ("analyzer", "skill_gap") else 0.7,
            max_tokens=1500, stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield "data: " + json.dumps({"type": "token", "content": token}) + "\n\n"

        yield "data: " + json.dumps({
            "type": "done", "mode": intent.type,
            "sources": _collect_sources(chunks), "accuracy": avg_accuracy(chunks),
            "is_followup": intent.is_followup,
        }) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
