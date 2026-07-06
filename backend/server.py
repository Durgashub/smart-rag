"""
SmartRAG AI Backend — Stage 4

What's new vs Stage 3:
  1. Conversation memory  — history: list[{role, content}] in every /api/ask request
                            GPT receives full prior turns → follow-up questions work
  2. LLM intent classification — replaces all regex patterns with one GPT call
                            Understands any phrasing, returns structured intent
  3. Map-reduce for cross-doc — each document queried separately, answers merged
                            Eliminates cross-contamination between documents
  4. Answer verification  — second GPT call checks answer against retrieved chunks
                            Catches hallucinations before they reach the user

Memory design:
  - Frontend keeps the history array in localStorage
  - Sends last MAX_HISTORY_TURNS turns with every /api/ask request
  - Backend never stores history — stateless, scales horizontally
  - Each turn = {role: "user"|"assistant", content: "..."}
  - Context injected AFTER system prompt, BEFORE current question
"""

import sys
import uuid
import subprocess
import re
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from query import retrieve, _is_cross_document_question

load_dotenv()
client = OpenAI()

CHAT_MODEL       = "gpt-4.1-mini"
MAX_HISTORY_TURNS = 10   # keep last 10 Q&A pairs (~5k tokens max)


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GENERAL = """
You are SmartRAG AI — an intelligent assistant that answers questions ONLY
using the retrieved document context provided.

Rules:
1. Use ONLY the information in the retrieved context. Never use outside knowledge.
2. If the answer is not in the context: "I don't know based on the provided documents."
3. Always cite which document(s) your answer comes from.
4. If documents disagree, report the conflict rather than picking one.
5. Quote only short phrases; otherwise summarize in your own words.

Conversation awareness:
- You have access to the conversation history above.
- Use it to understand follow-up questions ("rewrite that", "make it shorter",
  "now compare it to the other resume").
- When a follow-up refers to something said earlier, find it in the history.
- If a follow-up is ambiguous, ask one clarifying question.

Citation format — end every answer with:
Sources:
- <document_name> (page X if known)
"""

SYSTEM_PROMPT_CROSS_DOC = """
You are SmartRAG AI — extracting and comparing information across multiple documents.

RULES:
1. Each document is clearly separated by === DOCUMENT: filename === markers.
2. Only use information from within each document's section.
3. NEVER mix facts between documents.
4. List one entry per document — never skip a document.
5. If information is not found in a document, write "Not found in [filename]".

CONVERSATION AWARENESS:
- Use the conversation history to understand follow-up questions.
- "Compare them now" refers to the documents being discussed.
- "Which one is better?" refers to the candidates/documents mentioned earlier.

REQUIRED OUTPUT FORMAT:
1. [Name/Title] — [filename]
   - [relevant info]
2. [Name/Title] — [filename]
   - [relevant info]

Sources:
- [Document 1]
- [Document 2]
"""

SYSTEM_PROMPT_RESUME = """
You are SmartRAG AI — an expert resume coach and ATS optimization specialist.

Capabilities:
- Analyze and score resumes (overall + per section out of 10)
- Rewrite sections using strong action verbs and quantified achievements
- Identify skill gaps vs a job description
- Generate tailored cover letters
- Suggest ATS-friendly keywords

Rules:
- Reference actual content from the resume — be specific
- Use STAR format for experience rewrites
- Quantify achievements wherever possible
- Use conversation history for follow-ups ("make it shorter", "redo the summary")
"""

SYSTEM_PROMPT_ANALYZER = """
You are SmartRAG AI — an expert ATS resume analyzer.

Always respond in this exact format:

OVERALL SCORE: [X/10]

SECTION SCORES:
- Professional Summary: [X/10]
- Work Experience: [X/10]
- Skills: [X/10]
- Education: [X/10]
- Projects/Certifications: [X/10]

STRENGTHS:
[3-5 specific strengths from the resume]

WEAKNESSES:
[3-5 specific areas to improve]

ATS KEYWORDS FOUND: [keywords]
ATS KEYWORDS MISSING: [suggested keywords]

TOP 3 RECOMMENDATIONS:
1. [Most impactful change]
2. [Second most impactful]
3. [Third most impactful]
"""

SYSTEM_PROMPT_COVER_LETTER = """
You are SmartRAG AI — an expert cover letter writer.

Format:
- Professional header
- Opening: Hook + role interest
- Body 1: Most relevant experience (from resume)
- Body 2: Key achievement + skills match
- Closing: Call to action + sign-off

Rules:
- Reference specific achievements and numbers from the resume
- Under 400 words, human tone not templated
- Use conversation history if user asks to adjust ("shorter", "more formal")
"""

SYSTEM_PROMPT_SKILL_GAP = """
You are SmartRAG AI — a career development specialist.

Always respond in this exact format:

MATCH SCORE: [X%] - [Brief assessment]

SKILLS YOU HAVE (matching):
[skills from resume that match]

SKILLS YOU'RE MISSING:
[required skills not in resume]

NICE-TO-HAVE:
[optional beneficial skills]

ACTION PLAN:
1. [Most important skill + how to get it]
2. [Second skill + how]
3. [Third skill + how]

ESTIMATED TIME TO BE COMPETITIVE: [X months]
"""


# ── Stage 4: LLM Intent Classification ───────────────────────────────────────

class Intent(BaseModel):
    type: str          # single_doc | cross_doc | identity | resume | analyzer |
                       # cover_letter | skill_gap | out_of_scope
    reasoning: str     # why this intent was chosen (for logging)
    is_followup: bool  # is this a follow-up to a previous answer?


def classify_intent(question: str, history: list[dict]) -> Intent:
    """
    Replace ALL regex patterns with a single GPT call.

    Returns structured intent that drives:
    - Which system prompt to use
    - Whether to use adaptive/map-reduce retrieval
    - Whether to skip HyDE/rewriting
    - Whether this is a follow-up (needs history context)

    Fallback: returns single_doc intent if classification fails,
    so the system always produces an answer.
    """
    # Build conversation summary for context
    recent = history[-4:] if history else []
    history_summary = ""
    if recent:
        lines = []
        for turn in recent:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "")[:200]
            lines.append(f"{role}: {content}")
        history_summary = "\nRecent conversation:\n" + "\n".join(lines)

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier for a document Q&A system. "
                        "Classify the user's question into exactly one of these types:\n\n"
                        "- single_doc: question about one specific document or person\n"
                        "- cross_doc: needs info from ALL uploaded documents "
                        "(list names, compare all, who are the candidates, summarize all, "
                        "which candidate, compare resumes, list all)\n"
                        "- identity: asking for personal info about ONE person "
                        "(my name, my email, my phone, my contact)\n"
                        "- resume: rewrite, optimize, improve, tailor a resume\n"
                        "- analyzer: score, rate, evaluate, grade a resume\n"
                        "- cover_letter: write a cover letter or application letter\n"
                        "- skill_gap: missing skills, qualification match, am I qualified\n"
                        "- out_of_scope: ONLY if the question has absolutely nothing "
                        "to do with documents (e.g. 'what is the weather?', "
                        "'who won the Super Bowl?', 'tell me a joke'). "
                        "If the question COULD be answered from ANY uploaded document, "
                        "classify it as single_doc — not out_of_scope. "
                        "When in doubt, default to single_doc.\n\n"
                        "IMPORTANT: 'who are the candidates', 'list all names', "
                        "'compare the resumes', 'which candidate should I hire', "
                        "'what are the hazards', 'what does this document say about X' "
                        "are NEVER out_of_scope — they are cross_doc or single_doc.\n\n"
                        "is_followup: true ONLY if the question uses pronouns like "
                        "'it', 'that', 'them', 'the previous', 'same one' referring "
                        "to a specific prior answer. Questions about ALL documents "
                        "are NOT follow-ups even if asked after another question.\n\n"
                        "Return ONLY valid JSON, no markdown:\n"
                        '{"type": "<intent>", "reasoning": "<one sentence why>", "is_followup": <true|false>}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}{history_summary}",
                },
            ],
            temperature=0.1,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        intent = Intent(
            type=data.get("type", "single_doc"),
            reasoning=data.get("reasoning", ""),
            is_followup=data.get("is_followup", False),
        )
        print(f"  [Intent] {intent.type} | followup={intent.is_followup} | {intent.reasoning}")
        return intent
    except Exception as e:
        print(f"  [Intent] Classification failed: {e} — defaulting to single_doc")
        return Intent(type="single_doc", reasoning="fallback", is_followup=False)


# ── Stage 4: Map-reduce for cross-document questions ─────────────────────────

def simple_retrieve(question: str, session_id: str, top_k: int = 20) -> list[dict]:
    """
    Simple direct retrieval — bypasses ALL Stage 3 enhancements.
    No rewriting, no HyDE, no multi-query, no MMR, no cross-encoder.
    Just: embed question → FAISS search → return chunks.
    Used by map-reduce map phase to avoid HyDE hallucinating names.
    """
    import json as _json
    import numpy as np
    import faiss as _faiss

    store_dir = f"vector_store/{session_id}"
    try:
        index = _faiss.read_index(f"{store_dir}/index.faiss")
        with open(f"{store_dir}/metadata.json", "r", encoding="utf-8") as mf:
            metadata = _json.load(mf)
    except Exception as e:
        print(f"    [SimpleRetrieve] Failed to load index: {e}")
        return []

    response = client.embeddings.create(model="text-embedding-3-small", input=[question])
    query_vec = np.array([response.data[0].embedding], dtype="float32")

    k = min(top_k, len(metadata))
    distances, indices = index.search(query_vec, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        results.append({**metadata[idx], "distance": float(dist)})
    return results


def get_doc_chunks(session_id: str, filename: str, question: str, top_k: int) -> list[dict]:
    """
    Retrieve chunks belonging to ONE specific document.

    Three-level strategy to guarantee content from every document
    regardless of how vague the question is:

    Level 1 — Question-based retrieval:
      Run the actual question against the full index, take only chunks
      from this document. Works well for specific questions.

    Level 2 — Name-based retrieval:
      Search using the person's name (extracted from filename). This
      reliably finds the header/name chunk even for vague questions,
      because the name appears in every section of their resume.

    Level 3 — Metadata scan (guaranteed fallback):
      Read metadata.json directly and return the first N chunks from
      this document. Always succeeds as long as the doc was ingested.
    """
    import json as _json

    # Level 1: question-based (simple retrieval — no HyDE/rewrite/MMR)
    pool = simple_retrieve(question, session_id=session_id, top_k=top_k * 6)
    chunks = [c for c in pool if c.get("source") == filename]
    if chunks:
        print(f"    [Map] {filename} → {len(chunks)} chunks (L1 question-based)")
        return chunks[:top_k]

    # Level 2: name-based (filename without extension, underscores → spaces)
    name_query = re.sub(r'[_\.\-]', ' ', filename.rsplit('.', 1)[0]).strip()
    pool2 = simple_retrieve(name_query, session_id=session_id, top_k=top_k * 4)
    chunks2 = [c for c in pool2 if c.get("source") == filename]
    if chunks2:
        print(f"    [Map] {filename} → {len(chunks2)} chunks (L2 name-based)")
        return chunks2[:top_k]

    # Level 3: direct metadata scan — always works
    try:
        store_dir = f"vector_store/{session_id}"
        with open(f"{store_dir}/metadata.json", "r", encoding="utf-8") as mf:
            metadata = _json.load(mf)
        chunks3 = [m for m in metadata if m.get("source") == filename][:top_k]
        if chunks3:
            print(f"    [Map] {filename} → {len(chunks3)} chunks (L3 metadata scan)")
            return chunks3
    except Exception as e:
        print(f"    [Map] {filename} → metadata scan failed: {e}")

    print(f"    [Map] {filename} → 0 chunks (all levels failed)")
    return []


def map_reduce_answer(
    question: str,
    session_id: str,
    history: list[dict],
    top_k: int = 8,
) -> dict:
    """
    Map-reduce for cross-document questions.

    MAP:    For each uploaded document, retrieve chunks from THAT
            document only and generate a per-document answer in
            isolation — no other document's content is present.

    REDUCE: Merge all per-document answers into one final response.

    This makes cross-contamination structurally impossible.
    Each map GPT call sees only one document's content.
    """
    docs_dir = Path(f"docs/{session_id}")
    if not docs_dir.exists():
        return {"answer": "No documents found.", "sources": [],
                "chunks": [], "mode": "cross_doc", "accuracy": 0}

    uploaded_files = [f.name for f in sorted(docs_dir.iterdir()) if f.is_file()]
    if not uploaded_files:
        return {"answer": "No documents uploaded.", "sources": [],
                "chunks": [], "mode": "cross_doc", "accuracy": 0}

    print(f"\n  [MapReduce] {len(uploaded_files)} document(s) | question: '{question[:60]}'")

    # ── MAP phase ─────────────────────────────────────────────────────────────
    per_doc_answers = []
    all_chunks      = []

    for filename in uploaded_files:
        doc_chunks = get_doc_chunks(session_id, filename, question, top_k)

        if not doc_chunks:
            per_doc_answers.append({
                "filename": filename,
                "answer":   f"[{filename}]: Could not retrieve content.",
            })
            continue

        all_chunks.extend(doc_chunks)
        context = "\n\n".join(c["text"] for c in doc_chunks)

        map_system = (
            f"You are reading ONE document: {filename}\n\n"
            f"RULES:\n"
            f"1. Use ONLY the content provided below — nothing else.\n"
            f"2. The document belongs to one person. Find their name at the very top "
            f"   (before email, phone, or any section heading).\n"
            f"3. Answer the question as fully as possible from this document.\n"
            f"4. If specific information is missing, say what IS present instead of 'Not found'.\n"
            f"5. Be concise — 3-8 sentences maximum."
        )

        map_messages = [
            {"role": "system", "content": map_system},
            {"role": "user",   "content": f"Document content:\n{context}\n\nQuestion: {question}"},
        ]

        try:
            map_resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=map_messages,
                temperature=0.3,
                max_tokens=600,
            )
            doc_answer = map_resp.choices[0].message.content.strip()
            print(f"    [Map] '{filename}' answered ({len(doc_answer)} chars)")
        except Exception as e:
            doc_answer = f"Error processing {filename}: {e}"

        per_doc_answers.append({"filename": filename, "answer": doc_answer})

    # ── REDUCE phase ──────────────────────────────────────────────────────────
    if len(per_doc_answers) == 1:
        final_answer = per_doc_answers[0]["answer"]
    else:
        reduce_context = "\n\n".join(
            f"=== {d['filename']} ===\n{d['answer']}"
            for d in per_doc_answers
        )

        reduce_system = (
            "You are merging per-document answers into one final response.\n\n"
            "RULES:\n"
            "1. Each document's answer is under its === filename === header.\n"
            "2. Combine them into a clear numbered list — one entry per document.\n"
            "3. Do NOT add information not present in the per-document answers.\n"
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
            reduce_resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=reduce_messages,
                temperature=0.4,
                max_tokens=1500,
            )
            final_answer = reduce_resp.choices[0].message.content.strip()
            print(f"    [Reduce] Final answer: {len(final_answer)} chars")
        except Exception as e:
            # Fallback: concatenate map answers
            final_answer = "\n\n".join(
                f"**{d['filename']}:**\n{d['answer']}"
                for d in per_doc_answers
            )
            print(f"    [Reduce] Failed ({e}), using concatenation fallback")

    # Deduplicate chunks for display
    seen, unique_chunks = set(), []
    for c in all_chunks:
        key = (c.get("source"), c.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    return {
        "answer":   final_answer,
        "mode":     "cross_doc",
        "sources":  uploaded_files,
        "accuracy": avg_accuracy(unique_chunks) if unique_chunks else 55,
        "chunks": [
            {
                "source":   c.get("source", "Unknown"),
                "text":     c["text"][:300],
                "distance": c.get("distance"),
                "accuracy": calculate_accuracy(
                    c.get("distance", 1.0),
                    c.get("cross_encoder_score")
                ),
            }
            for c in unique_chunks[:8]
        ],
    }


# ── Stage 4: Answer verification ─────────────────────────────────────────────

def verify_answer(question: str, answer: str, chunks: list[dict]) -> tuple[str, bool]:
    """
    Check if the generated answer is actually supported by retrieved chunks.

    Returns (verified_answer, was_modified).
    If the answer contains unsupported claims, they are flagged or removed.

    Only runs for general and resume modes — not for cross_doc (map-reduce
    already isolates per-document) or out_of_scope.
    """
    context = "\n\n".join(c["text"][:400] for c in chunks[:5])

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an answer verifier for a RAG system. "
                        "Check if the answer is fully supported by the context.\n\n"
                        "If the answer contains claims NOT in the context, "
                        "remove or correct those claims.\n"
                        "If the answer is fully supported, return it unchanged.\n"
                        "Return ONLY the (possibly corrected) answer — no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer to verify:\n{answer}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        verified = response.choices[0].message.content.strip()
        was_modified = verified != answer
        if was_modified:
            print(f"  [Verify] Answer modified — unsupported claims removed")
        else:
            print(f"  [Verify] Answer verified — fully supported")
        return verified, was_modified
    except Exception as e:
        print(f"  [Verify] Failed: {e} — returning original answer")
        return answer, False


# ── Conversation-aware message builder ───────────────────────────────────────

def build_messages_with_history(
    system_prompt: str,
    context: str,
    question: str,
    history: list[dict],
) -> list[dict]:
    """
    Build the full messages array for a GPT call with conversation history.

    Structure:
    [
      {role: system, content: SYSTEM_PROMPT},
      {role: user,   content: Q1 + context1},   ← turn 1
      {role: assistant, content: A1},
      {role: user,   content: Q2 + context2},   ← turn 2
      ...
      {role: user,   content: CURRENT_Q + context}  ← current turn
    ]

    History turns are included as-is (the context they used is gone,
    but GPT can still understand references like "that section" or "the first candidate").
    The CURRENT question always includes fresh retrieved context.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Add history turns (capped at MAX_HISTORY_TURNS)
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]  # *2 because each turn = user + assistant
    for turn in recent_history:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })

    # Add current question with fresh context
    messages.append({
        "role": "user",
        "content": f"Context from your documents:\n{context}\n\nQuestion: {question}",
    })

    return messages


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_top_k(intent_type: str, question: str) -> int:
    if intent_type in ("analyzer", "skill_gap"):
        return 12
    if intent_type == "cross_doc":
        return 8  # per-document in map-reduce
    if len(question.split()) <= 5:
        return 12
    return 8


def get_system_prompt(intent_type: str) -> str:
    return {
        "analyzer":     SYSTEM_PROMPT_ANALYZER,
        "cover_letter": SYSTEM_PROMPT_COVER_LETTER,
        "skill_gap":    SYSTEM_PROMPT_SKILL_GAP,
        "resume":       SYSTEM_PROMPT_RESUME,
        "cross_doc":    SYSTEM_PROMPT_CROSS_DOC,
        "identity":     SYSTEM_PROMPT_GENERAL,
        "single_doc":   SYSTEM_PROMPT_GENERAL,
        "general":      SYSTEM_PROMPT_GENERAL,
        "out_of_scope": SYSTEM_PROMPT_GENERAL,
    }.get(intent_type, SYSTEM_PROMPT_GENERAL)


def build_context_prompt(question: str, chunks: list[dict], intent_type: str) -> str:
    """Build the context string sent to GPT for non-cross-doc questions."""
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    if intent_type == "analyzer":
        return f"Resume Content:\n{context}\n\nTask: {question}\n\nAnalyze thoroughly."
    if intent_type == "cover_letter":
        return f"Resume Content:\n{context}\n\nTask: {question}\n\nWrite a compelling cover letter."
    if intent_type == "skill_gap":
        return f"Documents:\n{context}\n\nTask: {question}\n\nAnalyze skill gap."
    if intent_type == "resume":
        return f"Documents:\n{context}\n\nTask: {question}\n\nOptimize based on content."
    return context  # context is injected via build_messages_with_history


def calculate_accuracy(distance: float, cross_encoder_score: float = None) -> int:
    CROSS_ENCODER_NOISE_FLOOR = -5.0
    if cross_encoder_score is not None and cross_encoder_score > CROSS_ENCODER_NOISE_FLOOR:
        normalized = (cross_encoder_score - CROSS_ENCODER_NOISE_FLOOR) / (10.0 - CROSS_ENCODER_NOISE_FLOOR)
        return round(min(100, max(50, normalized * 50 + 50)))
    min_dist, max_dist = 0.3, 1.5
    clamped = max(min_dist, min(max_dist, distance))
    return round(92 - ((clamped - min_dist) / (max_dist - min_dist)) * 46)


def avg_accuracy(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    return max(
        calculate_accuracy(c.get("distance", 1.0), c.get("cross_encoder_score"))
        for c in chunks
    )


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SmartRAG AI — Stage 4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://chat-smart.lovable.app",
    ],
    allow_origin_regex=r"https://.*\.lovable\.app|https://.*\.lovableproject\.com",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "x-session-id"],
    expose_headers=["x-session-id"],
)


# ── Request models ────────────────────────────────────────────────────────────

class HistoryTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = None
    history: list[HistoryTurn] = []  # ← NEW: conversation history


def get_session_dirs(session_id: str):
    docs_dir  = Path(f"docs/{session_id}")
    store_dir = Path(f"vector_store/{session_id}")
    docs_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir, store_dir


def get_files_for_session(session_id: str):
    docs_dir = Path(f"docs/{session_id}")
    if not docs_dir.exists():
        return []
    return [
        {"name": f.name, "size": f.stat().st_size}
        for f in docs_dir.iterdir()
        if f.is_file()
    ]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "stage": 4}


@app.get("/api/session")
def create_session(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
    return {"session_id": x_session_id}


@app.get("/")
def home():
    return {"message": "SmartRAG AI Backend — Stage 4"}


@app.get("/api/files")
def list_files(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        return {"files": []}
    return {"files": get_files_for_session(x_session_id)}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None),
):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    docs_dir, _ = get_session_dirs(x_session_id)
    destination = docs_dir / file.filename
    with open(destination, "wb") as f:
        f.write(await file.read())
    subprocess.run([sys.executable, "ingest.py", "--session", x_session_id], check=True)
    return {"files": get_files_for_session(x_session_id)}


@app.delete("/api/files/{filename}")
def delete_file(filename: str, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    docs_dir, _ = get_session_dirs(x_session_id)
    file_path = docs_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    file_path.unlink()
    remaining = get_files_for_session(x_session_id)
    if remaining:
        subprocess.run([sys.executable, "ingest.py", "--session", x_session_id], check=True)
    else:
        store_dir = Path(f"vector_store/{x_session_id}")
        if store_dir.exists():
            for f in store_dir.iterdir():
                f.unlink()
    return {"message": f"{filename} deleted.", "files": get_files_for_session(x_session_id)}


@app.post("/api/ask")
def ask_question(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None),
):
    """
    Main chat endpoint — Stage 4 pipeline:

    1. Classify intent (GPT call)
    2. If cross_doc → map-reduce (one GPT call per document, then merge)
    3. If out_of_scope → refuse without retrieval
    4. Otherwise → retrieve → build messages with history → GPT answer
    5. Verify answer (GPT call) for single_doc/identity
    6. Return answer + sources + accuracy + intent metadata
    """
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")

    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        return {
            "answer":   "No documents indexed yet. Please upload a document first.",
            "sources":  [], "chunks": [], "mode": "none", "accuracy": 0,
        }

    # Convert history pydantic models to plain dicts
    history = [{"role": t.role, "content": t.content} for t in request.history]

    # ── Step 1: Classify intent ──────────────────────────────────────────────
    intent = classify_intent(request.question, history)

    # Allow frontend to override intent (e.g. explicit mode button)
    if request.mode and request.mode != "auto":
        intent.type = request.mode
        print(f"  [Intent] Overridden by frontend: {intent.type}")

    # ── Step 2: Handle out_of_scope ─────────────────────────────────────────
    if intent.type == "out_of_scope":
        return {
            "answer":   "I can only answer questions about your uploaded documents. "
                        "That question appears to be outside the scope of your documents.",
            "sources":  [], "chunks": [], "mode": "out_of_scope", "accuracy": 0,
        }

    # ── Step 3: Cross-doc → map-reduce ──────────────────────────────────────
    if intent.type == "cross_doc":
        return map_reduce_answer(
            question=request.question,
            session_id=x_session_id,
            history=history,
            top_k=get_top_k(intent.type, request.question),
        )

    # ── Step 4: Standard retrieval + history-aware GPT call ─────────────────
    top_k  = get_top_k(intent.type, request.question)
    chunks = retrieve(request.question, session_id=x_session_id, top_k=top_k)

    if not chunks:
        return {
            "answer":   "No relevant content found in your documents.",
            "sources":  [], "chunks": [], "mode": intent.type, "accuracy": 0,
        }

    system_prompt   = get_system_prompt(intent.type)
    context_content = build_context_prompt(request.question, chunks, intent.type)

    messages = build_messages_with_history(
        system_prompt=system_prompt,
        context=context_content,
        question=request.question,
        history=history,
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3 if intent.type in ("analyzer", "skill_gap") else 0.7,
        max_tokens=1500,
    )
    answer = response.choices[0].message.content.strip()

    # ── Step 5: Verify answer (single_doc and identity only) ─────────────────
    if intent.type in ("single_doc", "identity", "general"):
        answer, _ = verify_answer(request.question, answer, chunks)

    # ── Step 6: Build response ───────────────────────────────────────────────
    sources_seen, enriched_sources = set(), []
    for c in chunks:
        src = c.get("source", "Unknown")
        if src not in sources_seen:
            sources_seen.add(src)
            enriched_sources.append(src)

    return {
        "answer":    answer,
        "mode":      intent.type,
        "is_followup": intent.is_followup,
        "sources":   enriched_sources,
        "accuracy":  avg_accuracy(chunks),
        "chunks": [
            {
                "source":   c.get("source", "Unknown"),
                "text":     c["text"][:300],
                "distance": c.get("distance"),
                "accuracy": calculate_accuracy(c.get("distance", 1.0), c.get("cross_encoder_score")),
            }
            for c in chunks
        ],
    }


# ── Streaming endpoint ───────────────────────────────────────────────────────

@app.post("/api/ask/stream")
async def ask_question_stream(
    request: QuestionRequest,
    x_session_id: Optional[str] = Header(None),
):
    """
    Streaming version of /api/ask.

    Returns a Server-Sent Events (SSE) stream so words appear one by one
    as GPT generates them — exactly like ChatGPT.

    SSE event format:
      data: {"type": "token", "content": "word "}


      data: {"type": "token", "content": "by "}


      data: {"type": "token", "content": "word"}


      data: {"type": "done", "mode": "...", "sources": [...],
             "accuracy": 84, "is_followup": false}



    The frontend reads tokens and appends them to the message bubble.
    The final "done" event carries all metadata (mode, sources, accuracy).

    Cross-doc (map-reduce) streams the reduce phase.
    All other intents stream the final GPT answer directly.
    """
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")

    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        async def no_docs():
            import json as _j
            yield "data: " + _j.dumps({"type": "token", "content": "No documents indexed yet. Please upload a document first."}) + "\n\n"
            yield "data: " + _j.dumps({"type": "done", "mode": "none", "sources": [], "accuracy": 0, "is_followup": False}) + "\n\n"
        return StreamingResponse(no_docs(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    history = [{"role": t.role, "content": t.content} for t in request.history]

    async def generate():
        import json

        # ── Step 1: Classify intent ──────────────────────────────────────────
        intent = classify_intent(request.question, history)
        if request.mode and request.mode != "auto":
            intent.type = request.mode

        # ── Step 2: Out of scope ─────────────────────────────────────────────
        if intent.type == "out_of_scope":
            msg = "I can only answer questions about your uploaded documents."
            yield "data: " + json.dumps({"type": "token", "content": msg}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "mode": "out_of_scope", "sources": [], "accuracy": 0, "is_followup": False}) + "\n\n"
            return

        # ── Step 3: Cross-doc → map-reduce then stream the reduce phase ──────
        if intent.type == "cross_doc":
            # Run map phase (non-streaming, fast per-doc answers)
            docs_dir = Path(f"docs/{x_session_id}")
            uploaded_files = [f.name for f in sorted(docs_dir.iterdir()) if f.is_file()] if docs_dir.exists() else []

            per_doc_answers = []
            all_chunks = []

            for filename in uploaded_files:
                doc_chunks = get_doc_chunks(x_session_id, filename, request.question, get_top_k(intent.type, request.question))
                if not doc_chunks:
                    per_doc_answers.append({"filename": filename, "answer": f"[{filename}]: Could not retrieve content."})
                    continue
                all_chunks.extend(doc_chunks)
                context = "\n\n".join(c["text"] for c in doc_chunks)
                map_messages = [
                    {"role": "system", "content": (
                        f"You are reading ONE document: {filename}. "
                        f"Answer ONLY from the content below. Be concise."
                    )},
                    {"role": "user", "content": f"Document content:\n{context}\n\nQuestion: {request.question}"},
                ]
                try:
                    map_resp = client.chat.completions.create(
                        model=CHAT_MODEL, messages=map_messages, temperature=0.3, max_tokens=600
                    )
                    per_doc_answers.append({"filename": filename, "answer": map_resp.choices[0].message.content.strip()})
                except Exception as e:
                    per_doc_answers.append({"filename": filename, "answer": f"Error: {e}"})

            # Stream the reduce phase
            reduce_context = "\n\n".join(f"=== {d['filename']} ===\n{d['answer']}" for d in per_doc_answers)
            reduce_messages = build_messages_with_history(
                system_prompt=(
                    "Merge these per-document answers into one final response. "
                    "List one entry per document. Do not skip any document."
                ),
                context=reduce_context,
                question=f"Merge for: {request.question}",
                history=history,
            )

            stream = client.chat.completions.create(
                model=CHAT_MODEL, messages=reduce_messages,
                temperature=0.4, max_tokens=1500, stream=True
            )
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield "data: " + json.dumps({"type": "token", "content": token}) + "\n\n"

            accuracy = avg_accuracy(all_chunks) if all_chunks else 55
            yield "data: " + json.dumps({"type": "done", "mode": "cross_doc", "sources": uploaded_files, "accuracy": accuracy, "is_followup": intent.is_followup}) + "\n\n"
            return

        # ── Step 4: Standard retrieval + streaming answer ────────────────────
        top_k  = get_top_k(intent.type, request.question)
        chunks = retrieve(request.question, session_id=x_session_id, top_k=top_k)

        if not chunks:
            msg = "No relevant content found in your documents."
            yield "data: " + json.dumps({"type": "token", "content": msg}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "mode": intent.type, "sources": [], "accuracy": 0, "is_followup": False}) + "\n\n"
            return

        system_prompt   = get_system_prompt(intent.type)
        context_content = build_context_prompt(request.question, chunks, intent.type)
        messages = build_messages_with_history(
            system_prompt=system_prompt,
            context=context_content,
            question=request.question,
            history=history,
        )

        # Stream the answer token by token
        full_answer = ""
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3 if intent.type in ("analyzer", "skill_gap") else 0.7,
            max_tokens=1500,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_answer += token
                yield "data: " + json.dumps({"type": "token", "content": token}) + "\n\n"

        # After streaming completes, collect metadata
        sources_seen, enriched_sources = set(), []
        for c in chunks:
            src = c.get("source", "Unknown")
            if src not in sources_seen:
                sources_seen.add(src)
                enriched_sources.append(src)

        yield "data: " + json.dumps({"type": "done", "mode": intent.type, "sources": enriched_sources, "accuracy": avg_accuracy(chunks), "is_followup": intent.is_followup}) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Dedicated endpoints (kept for backward compatibility) ────────────────────

@app.post("/api/analyze")
def analyze_resume(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    question = "Analyze and score my resume. Provide detailed feedback on all sections."
    chunks = retrieve(question, session_id=x_session_id, top_k=15)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not retrieve resume content.")
    messages = build_messages_with_history(
        system_prompt=SYSTEM_PROMPT_ANALYZER,
        context="\n\n".join(c["text"] for c in chunks),
        question=question,
        history=[],
    )
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    return {"analysis": response.choices[0].message.content, "accuracy": avg_accuracy(chunks), "chunks_used": len(chunks)}


@app.post("/api/cover-letter")
def generate_cover_letter(request: QuestionRequest, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    chunks = retrieve(request.question, session_id=x_session_id, top_k=12)
    history = [{"role": t.role, "content": t.content} for t in request.history]
    messages = build_messages_with_history(
        system_prompt=SYSTEM_PROMPT_COVER_LETTER,
        context="\n\n".join(c["text"] for c in chunks),
        question=request.question,
        history=history,
    )
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.7)
    return {"cover_letter": response.choices[0].message.content, "chunks_used": len(chunks)}


@app.post("/api/skill-gap")
def analyze_skill_gap(request: QuestionRequest, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    chunks = retrieve(request.question, session_id=x_session_id, top_k=12)
    history = [{"role": t.role, "content": t.content} for t in request.history]
    messages = build_messages_with_history(
        system_prompt=SYSTEM_PROMPT_SKILL_GAP,
        context="\n\n".join(c["text"] for c in chunks),
        question=request.question,
        history=history,
    )
    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3)
    return {"skill_gap": response.choices[0].message.content, "chunks_used": len(chunks)}


@app.get("/api/suggestions")
def get_suggestions(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        return {"suggestions": []}
    docs_dir = Path(f"docs/{x_session_id}")
    if not docs_dir.exists():
        return {"suggestions": []}
    files = [f for f in docs_dir.iterdir() if f.is_file()]
    if not files:
        return {"suggestions": []}

    file_previews = []
    for f in files[:3]:
        ext = f.suffix.lower()
        preview = ""
        try:
            if ext == ".pdf":
                import pdfplumber
                with pdfplumber.open(str(f)) as pdf:
                    if pdf.pages:
                        preview = (pdf.pages[0].extract_text() or "")[:800]
            elif ext == ".docx":
                from docx import Document
                doc = Document(str(f))
                preview = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])[:800]
            elif ext in {".txt", ".md", ".csv"}:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    preview = fh.read()[:800]
        except Exception:
            preview = f.name
        file_previews.append({"name": f.name, "preview": preview.strip()})

    file_names = [fp["name"] for fp in file_previews]
    num_files  = len(file_names)
    file_context = "\n\n".join([f"File: {fp['name']}\nPreview:\n{fp['preview']}" for fp in file_previews])

    if num_files == 1:
        instruction = f"Generate 3 specific questions for: {file_names[0]}"
    else:
        instruction = f"Generate 3 questions for {num_files} files: {', '.join(file_names)}. Include 1 comparison question."

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Generate specific suggested questions. Return ONLY a JSON array of 3 strings."},
                {"role": "user",   "content": f"{instruction}\n\n{file_context}\n\nReturn: [\"Q1?\", \"Q2?\", \"Q3?\"]"},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            suggestions = [s for s in suggestions if isinstance(s, str)][:3]
        else:
            suggestions = []
    except Exception:
        suggestions = [
            f"Summarize {file_names[0]}",
            "What are the main skills mentioned?",
            "Compare the candidates" if num_files > 1 else "What is the work experience?",
        ]

    return {"suggestions": suggestions}