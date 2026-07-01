import sys
import uuid
import subprocess
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Cookie, Response
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
- Match tone to the job description if provided
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
    """Detect which mode to use based on the question."""
    q = question.lower()

    # Resume analyzer
    if any(word in q for word in [
        "score", "rate my resume", "review my resume", "analyze my resume",
        "how good is", "rate it", "grade my", "evaluate my resume", "ats score"
    ]):
        return "analyzer"

    # Cover letter
    if any(word in q for word in [
        "cover letter", "write a letter", "application letter", "motivational letter"
    ]):
        return "cover_letter"

    # Skill gap
    if any(word in q for word in [
        "skill gap", "missing skills", "what skills", "skills i need",
        "am i qualified", "do i have", "match this job", "fit for this role",
        "missing for", "lack", "what am i missing"
    ]):
        return "skill_gap"

    # Resume optimization
    if any(word in q for word in [
        "rewrite", "improve", "optimize", "enhance", "update", "fix",
        "make it better", "stronger", "rephrase", "tailor", "customize",
        "job description", "ats", "keywords", "action verbs"
    ]):
        return "resume"

    # Default general mode
    return "general"


def get_system_prompt(mode: str) -> str:
    return {
        "analyzer": SYSTEM_PROMPT_ANALYZER,
        "cover_letter": SYSTEM_PROMPT_COVER_LETTER,
        "skill_gap": SYSTEM_PROMPT_SKILL_GAP,
        "resume": SYSTEM_PROMPT_RESUME,
        "general": SYSTEM_PROMPT_GENERAL,
    }.get(mode, SYSTEM_PROMPT_GENERAL)


def build_prompt(question: str, chunks: list[dict], mode: str) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)

    if mode == "analyzer":
        return f"""Resume Content:
{context}

Task: {question}

Analyze this resume thoroughly and provide structured feedback in the exact format specified."""

    if mode == "cover_letter":
        return f"""Resume Content:
{context}

Task: {question}

Write a compelling cover letter based on this resume content."""

    if mode == "skill_gap":
        return f"""Resume/Documents Content:
{context}

Task: {question}

Analyze the skill gap and provide structured feedback in the exact format specified."""

    if mode == "resume":
        return f"""Resume/Documents Content:
{context}

Task: {question}

Optimize and improve based on the content above. Be specific and reference actual content."""

    # General mode
    return f"""Context:
{context}

Question: {question}

Answer using only the context above. Cite which source(s) you used."""


def calculate_accuracy(distance: float) -> int:
    """Convert FAISS L2 distance to accuracy percentage."""
    return max(0, min(100, round((1 - distance / 2) * 100)))


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="SmartRAG AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.lovable\.app|https://.*\.lovableproject\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    mode: Optional[str] = None  # optional override: "resume", "analyzer", "cover_letter", "skill_gap", "general"


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


# ── Session ───────────────────────────────────────────────────────────────────

@app.get("/api/session")
def create_session(response: Response, session_id: str = Cookie(None)):
    """Create or return existing session cookie."""
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="none",
            secure=True,
            max_age=86400,
        )
    return {"session_id": session_id}


# ── Files ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "SmartRAG AI Backend Running"}


@app.get("/api/files")
def list_files(session_id: str = Cookie(None)):
    if not session_id:
        return {"files": []}
    return {"files": get_files_for_session(session_id)}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Cookie(None),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="No session. Call /api/session first.")

    suffix = Path(file.filename).suffix.lower()

    docs_dir, _ = get_session_dirs(session_id)
    destination = docs_dir / file.filename

    with open(destination, "wb") as f:
        f.write(await file.read())

    subprocess.run(
        [sys.executable, "ingest.py", "--session", session_id],
        check=True,
    )

    return {"files": get_files_for_session(session_id)}


@app.delete("/api/files/{filename}")
def delete_file(filename: str, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="No session.")

    docs_dir, _ = get_session_dirs(session_id)
    file_path = docs_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    file_path.unlink()

    remaining = get_files_for_session(session_id)
    if remaining:
        subprocess.run(
            [sys.executable, "ingest.py", "--session", session_id],
            check=True,
        )
    else:
        store_dir = Path(f"vector_store/{session_id}")
        if store_dir.exists():
            for f in store_dir.iterdir():
                f.unlink()

    return {
        "message": f"{filename} deleted successfully.",
        "files": get_files_for_session(session_id),
    }


# ── Ask ───────────────────────────────────────────────────────────────────────

@app.post("/api/ask")
def ask_question(request: QuestionRequest, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="No session.")

    store_dir = Path(f"vector_store/{session_id}")
    if not (store_dir / "index.faiss").exists():
        return {
            "answer": "No documents indexed yet. Please upload a document first.",
            "sources": [],
            "chunks": [],
            "mode": "none",
            "accuracy": 0,
        }

    # Auto-detect mode unless overridden
    mode = request.mode if request.mode else detect_mode(request.question)

    # Use more chunks for analysis tasks
    top_k = 12 if mode in ["analyzer", "skill_gap"] else 8
    chunks = retrieve(request.question, session_id=session_id, top_k=top_k)

    print(f"\n========== MODE: {mode.upper()} ==========")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} | Source: {chunk.get('source')} | Distance: {chunk.get('distance', 'N/A'):.4f}")
        print(chunk["text"][:300])
        print("-" * 60)

    if not chunks:
        return {
            "answer": "No relevant content found in your documents.",
            "sources": [],
            "chunks": [],
            "mode": mode,
            "accuracy": 0,
        }

    system_prompt = get_system_prompt(mode)
    prompt = build_prompt(request.question, chunks, mode)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3 if mode == "analyzer" else 0.7,
    )

    answer = response.choices[0].message.content

    # Build enriched sources with accuracy scores
    sources_seen = set()
    enriched_sources = []
    for c in chunks:
        src = c.get("source", "Unknown")
        if src not in sources_seen:
            sources_seen.add(src)
            enriched_sources.append(src)

    # Calculate per-chunk accuracy and overall confidence
    chunk_accuracies = [calculate_accuracy(c.get("distance", 1.0)) for c in chunks]
    overall_accuracy = round(sum(chunk_accuracies) / len(chunk_accuracies)) if chunk_accuracies else 0

    return {
        "answer": answer,
        "mode": mode,
        "sources": enriched_sources,
        "accuracy": overall_accuracy,
        "chunks": [
            {
                "source": c.get("source", "Unknown"),
                "text": c["text"][:300],
                "distance": c.get("distance"),
                "accuracy": calculate_accuracy(c.get("distance", 1.0)),
            }
            for c in chunks
        ],
    }


# ── Resume specific endpoints ─────────────────────────────────────────────────

@app.post("/api/analyze")
def analyze_resume(session_id: str = Cookie(None)):
    """Dedicated endpoint to fully analyze the uploaded resume."""
    if not session_id:
        raise HTTPException(status_code=401, detail="No session.")

    store_dir = Path(f"vector_store/{session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    question = "Analyze and score my resume. Provide detailed feedback on all sections."
    chunks = retrieve(question, session_id=session_id, top_k=15)

    if not chunks:
        raise HTTPException(status_code=400, detail="Could not retrieve resume content.")

    prompt = build_prompt(question, chunks, "analyzer")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ANALYZER},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    chunk_accuracies = [calculate_accuracy(c.get("distance", 1.0)) for c in chunks]
    overall_accuracy = round(sum(chunk_accuracies) / len(chunk_accuracies)) if chunk_accuracies else 0

    return {
        "analysis": response.choices[0].message.content,
        "accuracy": overall_accuracy,
        "chunks_used": len(chunks),
    }


@app.post("/api/cover-letter")
def generate_cover_letter(request: QuestionRequest, session_id: str = Cookie(None)):
    """Generate a cover letter from the uploaded resume."""
    if not session_id:
        raise HTTPException(status_code=401, detail="No session.")

    store_dir = Path(f"vector_store/{session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    chunks = retrieve(request.question, session_id=session_id, top_k=12)
    prompt = build_prompt(request.question, chunks, "cover_letter")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_COVER_LETTER},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    return {
        "cover_letter": response.choices[0].message.content,
        "chunks_used": len(chunks),
    }


@app.post("/api/skill-gap")
def analyze_skill_gap(request: QuestionRequest, session_id: str = Cookie(None)):
    """Analyze skill gap between resume and target role/job description."""
    if not session_id:
        raise HTTPException(status_code=401, detail="No session.")

    store_dir = Path(f"vector_store/{session_id}")
    if not (store_dir / "index.faiss").exists():
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    chunks = retrieve(request.question, session_id=session_id, top_k=12)
    prompt = build_prompt(request.question, chunks, "skill_gap")

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SKILL_GAP},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return {
        "skill_gap": response.choices[0].message.content,
        "chunks_used": len(chunks),
    }


# ── Dynamic Suggestions ───────────────────────────────────────────────────────

@app.get("/api/suggestions")
def get_suggestions(session_id: str = Cookie(None)):
    """Generate dynamic suggested questions based on uploaded files.
    Reads file names and a small preview of content — never stores or logs it.
    Returns empty list if no files uploaded.
    """
    if not session_id:
        return {"suggestions": []}

    docs_dir = Path(f"docs/{session_id}")
    if not docs_dir.exists():
        return {"suggestions": []}

    files = [f for f in docs_dir.iterdir() if f.is_file()]
    if not files:
        return {"suggestions": []}

    # Read a small preview of each file for context (first 800 chars only)
    file_previews = []
    for f in files[:3]:  # max 3 files
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
            preview = f.name  # fallback to just filename

        file_previews.append({
            "name": f.name,
            "preview": preview.strip()
        })

    # Build context for GPT to generate suggestions
    file_context = "\n\n".join([
        f"File: {fp['name']}\nContent preview:\n{fp['preview']}"
        for fp in file_previews
    ])

    file_names = [fp["name"] for fp in file_previews]
    num_files = len(file_names)

    if num_files == 1:
        instruction = f"""The user uploaded 1 file: {file_names[0]}
Generate 3 specific, useful suggested questions a user would ask about this exact document.
Make them specific to the actual content — not generic like 'summarize this document'."""
    else:
        instruction = f"""The user uploaded {num_files} files: {', '.join(file_names)}
Generate 3 specific suggested questions. Include at least 1 comparison question between the files.
Make them specific to the actual content — not generic."""

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
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            suggestions = [s for s in suggestions if isinstance(s, str)][:3]
        else:
            suggestions = []
    except Exception as e:
        print(f"Suggestions generation failed: {e}")
        # Fallback to generic suggestions based on file count
        if num_files == 1:
            suggestions = [
                f"Summarize the key points of {file_names[0]}",
                "What are the main findings in this document?",
                "What skills or experience does this document highlight?"
            ]
        else:
            suggestions = [
                f"Compare {file_names[0]} and {file_names[1]}",
                "What are the key differences between these files?",
                "Summarize the most important information across all files"
            ]

    return {"suggestions": suggestions}