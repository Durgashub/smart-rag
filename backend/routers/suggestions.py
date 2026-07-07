"""
routers/suggestions.py — GET /api/suggestions

Reads file previews and asks GPT to generate 3 specific questions
the user would likely ask about those documents.
"""

import re
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Header

from config import client, CHAT_MODEL

router = APIRouter()


@router.get("/api/suggestions")
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
        ext     = f.suffix.lower()
        preview = ""
        try:
            if ext == ".pdf":
                import pdfplumber
                with pdfplumber.open(str(f)) as pdf:
                    if pdf.pages:
                        preview = (pdf.pages[0].extract_text() or "")[:800]
            elif ext == ".docx":
                from docx import Document
                doc     = Document(str(f))
                preview = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])[:800]
            elif ext in {".txt", ".md", ".csv"}:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    preview = fh.read()[:800]
        except Exception:
            preview = f.name
        file_previews.append({"name": f.name, "preview": preview.strip()})

    file_names   = [fp["name"] for fp in file_previews]
    num_files    = len(file_names)
    file_context = "\n\n".join(
        f"File: {fp['name']}\nPreview:\n{fp['preview']}" for fp in file_previews
    )
    instruction  = (
        f"Generate 3 specific questions for: {file_names[0]}"
        if num_files == 1
        else f"Generate 3 questions for {num_files} files: {', '.join(file_names)}. Include 1 comparison question."
    )

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
        raw         = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()
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
