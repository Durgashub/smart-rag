"""
routers/career.py — resume-specific endpoints.

  POST /api/analyze      → ATS resume scoring
  POST /api/cover-letter → cover letter generation
  POST /api/skill-gap    → skill gap analysis
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Header

from config import client, CHAT_MODEL
from schemas import QuestionRequest
from prompts import ANALYZER, COVER_LETTER, SKILL_GAP
from services.sessions import has_index
from services.generation import build_messages_with_history
from services.accuracy import avg_accuracy
from retrieval.pipeline import retrieve

router = APIRouter()


@router.post("/api/analyze")
def analyze_resume(x_session_id: Optional[str] = Header(None)):
    """Full ATS resume analysis with structured scoring (OVERALL SCORE, SECTION SCORES, etc.)"""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    if not has_index(x_session_id):
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    question = "Analyze and score my resume. Provide detailed feedback on all sections."
    chunks   = retrieve(question, session_id=x_session_id, top_k=15)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not retrieve resume content.")
    messages = build_messages_with_history(
        system_prompt=ANALYZER,
        context="\n\n".join(c["text"] for c in chunks),
        question=question,
        history=[],
    )
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return {"analysis": response.choices[0].message.content, "accuracy": avg_accuracy(chunks), "chunks_used": len(chunks)}


@router.post("/api/cover-letter")
def generate_cover_letter(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None),
):
    """Generate a tailored cover letter from the uploaded resume."""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    if not has_index(x_session_id):
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    chunks  = retrieve(request.question, session_id=x_session_id, top_k=12)
    history = [{"role": t.role, "content": t.content} for t in request.history]
    messages = build_messages_with_history(
        system_prompt=COVER_LETTER,
        context="\n\n".join(c["text"] for c in chunks),
        question=request.question,
        history=history,
    )
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.7)
    return {"cover_letter": response.choices[0].message.content, "chunks_used": len(chunks)}


@router.post("/api/skill-gap")
def analyze_skill_gap(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None),
):
    """Analyze skill gaps between the uploaded resume and a target role."""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    if not has_index(x_session_id):
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    chunks  = retrieve(request.question, session_id=x_session_id, top_k=12)
    history = [{"role": t.role, "content": t.content} for t in request.history]
    messages = build_messages_with_history(
        system_prompt=SKILL_GAP,
        context="\n\n".join(c["text"] for c in chunks),
        question=request.question,
        history=history,
    )
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3)
    return {"skill_gap": response.choices[0].message.content, "chunks_used": len(chunks)}
