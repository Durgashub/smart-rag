"""
Ingest documents for a specific user session.

Usage:
    python ingest.py --session <session_id>

Reads from:  docs/<session_id>/
Writes to:   vector_store/<session_id>/
"""

import os
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

load_dotenv()

client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_TEXT_LEN_BEFORE_OCR = 20

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
POPPLER_PATH = os.getenv("POPPLER_PATH") or None


def _table_to_markdown(table_rows) -> str:
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table_rows if row]
    if not rows:
        return ""
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _ocr_page(pdf_path: str, page_number: int) -> str:
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=200,
        poppler_path=POPPLER_PATH,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def load_text_from_pdf(path: str) -> str:
    page_texts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            tables = page.extract_tables()
            table_blocks = [_table_to_markdown(t) for t in tables if t]
            table_blocks = [t for t in table_blocks if t]
            combined = text
            if table_blocks:
                combined = (combined + "\n\n" + "\n\n".join(table_blocks)).strip()

            if len(combined) < MIN_TEXT_LEN_BEFORE_OCR:
                try:
                    ocr_text = _ocr_page(path, i)
                    if ocr_text.strip():
                        combined = ocr_text.strip()
                except Exception as e:
                    print(f"  Warning: OCR failed on page {i} of {Path(path).name}: {e}")

            if combined:
                page_texts.append(f"[Page {i}]\n{combined}")

    return "\n\n".join(page_texts)


def load_text_from_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def load_text_from_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return load_text_from_pdf(path)
    if path.lower().endswith(".docx"):
        return load_text_from_docx(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def is_low_value_chunk(text: str) -> bool:
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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end].strip())
                start += chunk_size - overlap
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            flush()
            current = para

    flush()
    return [c for c in chunks if c]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session ID to scope docs and vector store")
    args = parser.parse_args()

    session_id = args.session
    docs_dir = f"docs/{session_id}"
    store_dir = f"vector_store/{session_id}"

    os.makedirs(store_dir, exist_ok=True)

    file_paths = glob.glob(f"{docs_dir}/**/*.*", recursive=True)
    file_paths = [p for p in file_paths if p.lower().endswith((".txt", ".md", ".pdf"))]

    if not file_paths:
        print(f"No files found in '{docs_dir}/'. Nothing to index.")
        return

    all_chunks = []
    metadata = []
    skipped_count = 0

    for path in file_paths:
        print(f"Loading {path} ...")
        text = load_text_from_file(path)
        raw_chunks = chunk_text(text)
        chunks = [c for c in raw_chunks if not is_low_value_chunk(c)]
        skipped_count += len(raw_chunks) - len(chunks)
        print(f"  -> {len(chunks)} chunk(s) kept, {len(raw_chunks) - len(chunks)} skipped")
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"source": Path(path).name, "text": chunk})

    print(f"Total: {len(all_chunks)} chunks from {len(file_paths)} file(s) ({skipped_count} skipped).")
    print("Generating embeddings...")
    embeddings = embed_texts(all_chunks)

    embedding_matrix = np.array(embeddings, dtype="float32")
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    faiss.write_index(index, f"{store_dir}/index.faiss")
    with open(f"{store_dir}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Index saved to '{store_dir}/'.")


if __name__ == "__main__":
    main()