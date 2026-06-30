import sys
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from query import retrieve
from pathlib import Path
import subprocess


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

client = OpenAI()

CHAT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions using ONLY the
provided context. If the answer isn't in the context, say you don't know
rather than guessing.
"""

app = FastAPI(title="RAG Backend")

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.lovableproject\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return f"""Context:
{context}

Question:
{question}

Answer using only the context above. Cite which source(s) you used."""

def get_files():
    files = []

    for file in DOCS_DIR.iterdir():
        if file.is_file():
            files.append(
                {
                    "name": file.name,
                    "size": file.stat().st_size,
                }
            )

    return files

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "RAG Backend Running"}


@app.get("/api/files")
def list_files():
    return {"files": get_files()}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}

    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and MD files are allowed."
        )

    destination = DOCS_DIR / file.filename

    with open(destination, "wb") as f:
        f.write(await file.read())

    # rebuild vector index
    subprocess.run([sys.executable, "ingest.py"], check=True)

    return {"files": get_files()}

@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    file_path = DOCS_DIR / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found."
        )

    file_path.unlink()

    # Rebuild vector index after deleting a document
    subprocess.run([sys.executable, "ingest.py"], check=True)

    return {
        "message": f"{filename} deleted successfully.",
        "files": get_files()
    }

@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    chunks = retrieve(request.question, top_k=8)

    print("\n========== RETRIEVED CHUNKS ==========")

    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}")
        print(f"Source: {chunk.get('source', 'Unknown')}")
        print(f"Page: {chunk.get('page', 'Unknown')}")
        print(chunk["text"][:400])   # First 400 characters
        print("-" * 80)
        print(f"Distance: {chunk.get('distance', 'N/A')}")

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": []
        }

    # Rest of your code...

    

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "sources": []
        }
    


    
    

    prompt = build_prompt(request.question, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    sources = sorted(
        list(
            set(chunk["source"] for chunk in chunks)
        )
    )

    return {
        "answer": answer,
        "sources": sources
    }

