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
from fastapi.middleware.cors import CORSMiddleware

from query import retrieve

load_dotenv()
client = OpenAI()
CHAT_MODEL = "gpt-4.1-mini"

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GENERAL = """
You are a helpful assistant that answers questions using ONLY the
provided context. If the answer isn't in the context, say you don't know
rather than guessing. Always cite which source(s) you used.

Important: For personal information questions (name, contact, identity),
look carefully at document headers, titles, and the beginning of the content.
The person's name is usually at the very top of a resume or document.
"""

SYSTEM_PROMPT_RESUME = """
You are an expert resume coach, career advisor, and ATS optimization specialist.
The user has uploaded their resume and possibly a job description.

Your capabilities:
- Analyze and score resumes (overall + per section out of 10)
- Rewrite weak sections using strong action verbs and quantified achievements
- Identify skill gaps between resume and job description
- Generate tailored cover letters based on resume content
- Suggest ATS-friendly keywords based on the target role
- Recommend certifications or skills to add

Rules:
- Always be specific — reference actual content from their resume
- Use STAR format (Situation, Task, Action, Result) for experience rewrites
- Quantify achievements wherever possible (e.g. "improved performance by 40%")
- If a job description is provided, optimize specifically for that role
- If something is not in the context, say so honestly
- Structure your response clearly with sections and bullet points
"""

SYSTEM_PROMPT_ANALYZER = """
You are an expert ATS (Applicant Tracking System) resume analyzer.
Score the resume and provide structured feedback.

Always respond in this exact format:

OVERALL SCORE: [X/10]

SECTION SCORES:
- Professional Summary: [X/10]
- Work Experience: [X/10]
- Skills: [X/10]
- Education: [X/10]
- Projects/Certifications: [X/10]

STRENGTHS:
[List 3-5 specific strengths from the resume]

WEAKNESSES:
[List 3-5 specific areas to improve]

ATS KEYWORDS FOUND: [list keywords]
ATS KEYWORDS MISSING (suggested): [list suggested keywords for the role]

TOP 3 RECOMMENDATIONS:
1. [Most impactful change]
2. [Second most impactful change]
3. [Third most impactful change]
"""

SYSTEM_PROMPT_COVER_LETTER = """
You are an expert cover letter writer.
Using the resume context provided, write a compelling, personalized cover letter.

Format:
- Professional header
- Opening paragraph: Hook + role interest
- Body paragraph 1: Most relevant experience (from resume)
- Body paragraph 2: Key achievement + skills match
- Closing paragraph: Call to action
- Professional sign-off

Rules:
- Reference specific achievements and numbers from the resume
- Keep it to 3-4 paragraphs, under 400 words
- Make it sound human, not templated
"""

SYSTEM_PROMPT_SKILL_GAP = """
You are a career development specialist analyzing skill gaps.
Compare the resume against the job description or target role.

Always respond in this exact format:

MATCH SCORE: [X%] - [Brief assessment]

SKILLS YOU HAVE (matching the role):
[List skills from resume that match]

SKILLS YOU'RE MISSING:
[List required skills not found in resume]

NICE-TO-HAVE SKILLS TO ADD:
[List optional but beneficial skills]

ACTION PLAN:
1. [Most important skill to acquire + how]
2. [Second skill + how]
3. [Third skill + how]

ESTIMATED TIME TO BE COMPETITIVE: [X months]
"""

# ── Mode detection ────────────────────────────────────────────────────────────

def detect_mode(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["score", "rate my resume", "review my resume",
        "analyze my resume", "how good is", "rate it", "grade my",
        "evaluate my resume", "ats score"]):
        return "analyzer"
    if any(w in q for w in ["cover letter", "write a letter",
        "application letter", "motivational letter"]):
        return "cover_letter"
    if any(w in q for w in ["skill gap", "missing skills", "what skills",
        "skills i need", "am i qualified", "do i have", "match this job",
        "fit for this role", "missing for", "lack", "what am i missing"]):
        return "skill_gap"
    if any(w in q for w in ["rewrite", "improve", "optimize", "enhance",
        "update", "fix", "make it better", "stronger", "rephrase", "tailor",
        "customize", "job description", "ats", "keywords", "action verbs"]):
        return "resume"
    return "general"


def get_top_k(question: str, mode: str) -> int:
    if mode in ["analyzer", "skill_gap"]:
        return 12
    if len(question.split()) <= 5:
        return 12
    return 8


def get_system_prompt(mode: str) -> str:
    return {
        "analyzer":     SYSTEM_PROMPT_ANALYZER,
        "cover_letter": SYSTEM_PROMPT_COVER_LETTER,
        "skill_gap":    SYSTEM_PROMPT_SKILL_GAP,
        "resume":       SYSTEM_PROMPT_RESUME,
        "general":      SYSTEM_PROMPT_GENERAL,
    }.get(mode, SYSTEM_PROMPT_GENERAL)


def build_prompt(question: str, chunks: list[dict], mode: str) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    if mode == "analyzer":
        return f"Resume Content:\n{context}\n\nTask: {question}\n\nAnalyze thoroughly and provide structured feedback."
    if mode == "cover_letter":
        return f"Resume Content:\n{context}\n\nTask: {question}\n\nWrite a compelling cover letter."
    if mode == "skill_gap":
        return f"Resume/Documents Content:\n{context}\n\nTask: {question}\n\nAnalyze skill gap with structured feedback."
    if mode == "resume":
        return f"Resume/Documents Content:\n{context}\n\nTask: {question}\n\nOptimize based on content above."
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above. Cite sources."


def calculate_accuracy(distance: float, cross_encoder_score: float = None) -> int:
    """
    Convert retrieval signals to an accuracy percentage shown in the UI.

    Strategy:
    - If cross-encoder score is available AND reasonable (> -5), use it.
      ms-marco-MiniLM-L-6-v2 on short text: relevant ≈ 0..+10, noise ≈ -10..-5
      On long parent chunks it often scores -8..-10 even for relevant content,
      so we only trust it when it's above the noise floor (-5).
    - Otherwise fall back to FAISS L2 distance, which is reliable regardless
      of chunk length.

    FAISS L2 distance for OpenAI text-embedding-3-small:
      0.3 → very close match   → ~92%
      0.6 → good match         → ~82%
      0.9 → moderate match     → ~70%
      1.2 → weak match         → ~58%
      1.5 → distant            → ~46%
    """
    CROSS_ENCODER_NOISE_FLOOR = -5.0

    if cross_encoder_score is not None and cross_encoder_score > CROSS_ENCODER_NOISE_FLOOR:
        # Score range -5..+10 → map to 50%..100%
        normalized = (cross_encoder_score - CROSS_ENCODER_NOISE_FLOOR) / (10.0 - CROSS_ENCODER_NOISE_FLOOR)
        return round(min(100, max(50, normalized * 50 + 50)))

    # FAISS L2 distance fallback
    min_dist = 0.3
    max_dist = 1.5
    clamped = max(min_dist, min(max_dist, distance))
    accuracy = 92 - ((clamped - min_dist) / (max_dist - min_dist)) * 46
    return round(accuracy)


def avg_accuracy(chunks: list[dict]) -> int:
    if not chunks:
        return 0
    scores = [
        calculate_accuracy(c.get("distance", 1.0), c.get("cross_encoder_score"))
        for c in chunks
    ]
    return round(sum(scores) / len(scores))


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="SmartRAG AI Backend")

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


class QuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = None


def get_session_dirs(session_id: str):
    docs_dir = Path(f"docs/{session_id}")
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
    return {"status": "ok"}


@app.get("/api/session")
def create_session(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
    return {"session_id": x_session_id}


@app.get("/")
def home():
    return {"message": "SmartRAG AI Backend Running"}


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
        raise HTTPException(status_code=401, detail="No session ID. Call /api/session first.")
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
    return {"message": f"{filename} deleted successfully.", "files": get_files_for_session(x_session_id)}


@app.post("/api/ask")
def ask_question(request: QuestionRequest, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        return {"answer": "No documents indexed yet. Please upload a document first.",
                "sources": [], "chunks": [], "mode": "none", "accuracy": 0}

    mode   = request.mode if request.mode else detect_mode(request.question)
    top_k  = get_top_k(request.question, mode)
    chunks = retrieve(request.question, session_id=x_session_id, top_k=top_k)

    if not chunks:
        return {"answer": "No relevant content found in your documents.",
                "sources": [], "chunks": [], "mode": mode, "accuracy": 0}

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt(mode)},
            {"role": "user",   "content": build_prompt(request.question, chunks, mode)},
        ],
        temperature=0.3 if mode == "analyzer" else 0.7,
    )

    sources_seen, enriched_sources = set(), []
    for c in chunks:
        src = c.get("source", "Unknown")
        if src not in sources_seen:
            sources_seen.add(src)
            enriched_sources.append(src)

    return {
        "answer":   response.choices[0].message.content,
        "mode":     mode,
        "sources":  enriched_sources,
        "accuracy": avg_accuracy(chunks),
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
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ANALYZER},
            {"role": "user",   "content": build_prompt(question, chunks, "analyzer")},
        ],
        temperature=0.2,
    )
    return {"analysis": response.choices[0].message.content,
            "accuracy": avg_accuracy(chunks), "chunks_used": len(chunks)}


@app.post("/api/cover-letter")
def generate_cover_letter(request: QuestionRequest, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    chunks = retrieve(request.question, session_id=x_session_id, top_k=12)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_COVER_LETTER},
            {"role": "user",   "content": build_prompt(request.question, chunks, "cover_letter")},
        ],
        temperature=0.7,
    )
    return {"cover_letter": response.choices[0].message.content, "chunks_used": len(chunks)}


@app.post("/api/skill-gap")
def analyze_skill_gap(request: QuestionRequest, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    store_dir = Path(f"vector_store/{x_session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")
    chunks = retrieve(request.question, session_id=x_session_id, top_k=12)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SKILL_GAP},
            {"role": "user",   "content": build_prompt(request.question, chunks, "skill_gap")},
        ],
        temperature=0.3,
    )
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
            elif ext in {".xlsx", ".xls"}:
                import openpyxl
                wb = openpyxl.load_workbook(str(f), data_only=True)
                ws = wb.active
                rows = []
                for row in ws.iter_rows(values_only=True, max_row=10):
                    rows.append(" | ".join(str(c) for c in row if c is not None))
                preview = "\n".join(rows)[:800]
        except Exception:
            preview = f.name
        file_previews.append({"name": f.name, "preview": preview.strip()})

    file_names = [fp["name"] for fp in file_previews]
    num_files  = len(file_names)
    file_context = "\n\n".join([
        f"File: {fp['name']}\nContent preview:\n{fp['preview']}"
        for fp in file_previews
    ])

    if num_files == 1:
        instruction = f"The user uploaded 1 file: {file_names[0]}\nGenerate 3 specific, useful suggested questions a user would ask about this exact document."
    else:
        instruction = f"The user uploaded {num_files} files: {', '.join(file_names)}\nGenerate 3 specific suggested questions. Include at least 1 comparison question between the files."

    prompt = f"""{instruction}

File content previews:
{file_context}

Rules:
- Return ONLY a JSON array of exactly 3 question strings
- No preamble, no explanation, no markdown — just the raw JSON array
- Questions must be specific to the actual content shown
- Keep each question under 12 words
- If files look like resumes: suggest resume-specific questions
- If files look like data/reports: suggest analysis questions
- If mixed file types: suggest comparison and analysis questions

Example format:
["Question 1?", "Question 2?", "Question 3?"]"""

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You generate specific, relevant suggested questions based on document content. Return only a JSON array."},
                {"role": "user",   "content": prompt},
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
    except Exception as e:
        print(f"Suggestions generation failed: {e}")
        if num_files == 1:
            suggestions = [
                f"Summarize the key points of {file_names[0]}",
                "What are the main findings in this document?",
                "What skills or experience does this document highlight?",
            ]
        else:
            suggestions = [
                f"Compare {file_names[0]} and {file_names[1]}",
                "What are the key differences between these files?",
                "Summarize the most important information across all files",
            ]

    return {"suggestions": suggestions}