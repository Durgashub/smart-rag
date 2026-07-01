import sys
import uuid
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware

from query import retrieve

load_dotenv()

client = OpenAI()

CHAT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using ONLY the
provided context. If the answer isn't in the context, say you don't know
rather than guessing.
"""

app = FastAPI(title="RAG Backend")

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


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return f"""Context:
{context}

Question:
{question}

Answer using only the context above. Cite which source(s) you used."""


# ── Session ──────────────────────────────────────────────────────────────────

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
            max_age=86400,  # 24 hours
        )
    return {"session_id": session_id}


# ── Files ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "RAG Backend Running"}


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
    
    allowed = None  # Allow all file types
    suffix = Path(file.filename).suffix.lower()

    docs_dir, _ = get_session_dirs(session_id)
    destination = docs_dir / file.filename

    with open(destination, "wb") as f:
        f.write(await file.read())

    # Rebuild vector index for this session only
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

    # Rebuild index (or clear it if no files remain)
    remaining = get_files_for_session(session_id)
    if remaining:
        subprocess.run(
            [sys.executable, "ingest.py", "--session", session_id],
            check=True,
        )
    else:
        # Clear the vector store for this session
        store_dir = Path(f"vector_store/{session_id}")
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

    # Check this session has an index
    store_dir = Path(f"vector_store/{session_id}")
    if not (store_dir / "index.faiss").exists():
        return {
            "answer": "No documents indexed yet. Please upload a PDF first.",
            "sources": [],
            "chunks": [],
        }

    chunks = retrieve(request.question, session_id=session_id, top_k=8)

    print("\n========== RETRIEVED CHUNKS ==========")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}")
        print(f"Source: {chunk.get('source', 'Unknown')}")
        print(f"Page: {chunk.get('page', 'Unknown')}")
        print(chunk["text"][:400])
        print("-" * 80)

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": [],
            "chunks": [],
        }

    prompt = build_prompt(request.question, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    answer = response.choices[0].message.content
    sources = sorted(set(chunk["source"] for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks": [
            {
                "source": c.get("source", "Unknown"),
                "text": c["text"][:300],
                "distance": c.get("distance"),
            }
            for c in chunks
        ],
    }