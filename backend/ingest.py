"""
Ingest documents from the docs/ folder:
- loads .txt and .md files directly
- loads .pdf files with pdfplumber, extracting tables as markdown and
  falling back to OCR for pages that have no selectable text (scanned pages)
- splits the result into paragraph-aware chunks (keeps tables and
  sentences intact far more often than fixed-width character slicing)
- embeds each chunk with OpenAI's embedding model
- stores chunks + embeddings in a local FAISS index

Run this whenever you add or change documents in docs/.

Requires Tesseract OCR and Poppler installed on the system for the OCR
fallback to work -- see README for Windows install steps. If they aren't
installed, normal (non-scanned) PDFs still work fine; only the OCR
fallback is skipped, with a warning printed.

Usage:
    python ingest.py
"""

import os
import re
import json
import glob
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

DOCS_DIR = "docs"
STORE_DIR = "vector_store"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200      # characters per chunk (bumped up to suit longer documents)
CHUNK_OVERLAP = 200    # overlap between consecutive chunks
MIN_TEXT_LEN_BEFORE_OCR = 20  # pages with less extracted text than this are treated as scanned

# Optional: point at non-PATH installs of Tesseract / Poppler (mainly for
# Windows -- set these in your .env file if they aren't on your PATH).
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
POPPLER_PATH = os.getenv("POPPLER_PATH") or None


def _table_to_markdown(table_rows) -> str:
    """Convert a pdfplumber-extracted table (list of row lists) into a
    markdown table string, so row/column structure survives chunking
    instead of collapsing into one run-on line of text."""
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table_rows if row]
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows[1:]:
        row = (row + [""] * len(header))[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _ocr_page(pdf_path: str, page_number: int) -> str:
    """Rasterize a single PDF page (1-indexed) and run OCR on it."""
    images = convert_from_path(
        pdf_path, first_page=page_number, last_page=page_number, dpi=200,
        poppler_path=POPPLER_PATH,
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def load_text_from_pdf(path: str) -> str:
    """Extract text page by page, pulling tables out as markdown and
    falling back to OCR for pages with no real text layer (scanned pages)."""
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
                    print("  (Is Tesseract/Poppler installed? See README.)")

            if combined:
                page_texts.append(f"[Page {i}]\n{combined}")

    return "\n\n".join(page_texts)


def load_text_from_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return load_text_from_pdf(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def is_low_value_chunk(text: str) -> bool:
    """Flag chunks that are pure noise for retrieval purposes: bare
    "CONCEPT CHECK" question stubs with no real content, or chunks
    dominated by numbered bibliography/footnote entries. These chunks
    can win a retrieval slot on keyword/semantic overlap alone (e.g. a
    citation block mentioning "Mintzberg" for a Mintzberg question)
    without containing any of the actual explanatory content -- which
    drags down context precision without helping the answer at all."""
    stripped = text.strip()
    if not stripped:
        return True

    lines = [l for l in stripped.splitlines() if l.strip()]
    if not lines:
        return True

    # Bare concept-check table: every line is a markdown table row, and
    # there's barely any non-table text alongside it.
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    if table_lines == len(lines) and "CONCEPT CHECK" in stripped.upper():
        return True

    # Reference/footnote-heavy chunk: most lines look like numbered
    # bibliography entries, e.g. "12. Mintzberg, H. (1973). The Nature..."
    numbered = sum(1 for l in lines if re.match(r"^\d{1,3}[\.\)]\s", l.strip()))
    if len(lines) >= 3 and numbered / len(lines) > 0.5:
        return True

    return False


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Paragraph-aware chunking: pack whole paragraphs together up to
    chunk_size, only splitting mid-paragraph when a single paragraph (e.g.
    a big table) is longer than chunk_size on its own. This keeps tables
    and sentences intact far more often than fixed-width character slicing."""
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
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


def main():
    os.makedirs(STORE_DIR, exist_ok=True)

    file_paths = glob.glob(f"{DOCS_DIR}/**/*.*", recursive=True)
    file_paths = [p for p in file_paths if p.lower().endswith((".txt", ".md", ".pdf"))]

    if not file_paths:
        print(f"No .txt, .md, or .pdf files found in '{DOCS_DIR}/'. Add some docs and re-run.")
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
        print(f"  -> {len(chunks)} chunk(s) kept, {len(raw_chunks) - len(chunks)} low-value chunk(s) skipped")
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"source": Path(path).name, "text": chunk})

    print(f"Created {len(all_chunks)} chunks from {len(file_paths)} file(s) ({skipped_count} skipped as low-value).")
    print("Generating embeddings (this calls the OpenAI API)...")
    embeddings = embed_texts(all_chunks)

    embedding_matrix = np.array(embeddings, dtype="float32")
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    faiss.write_index(index, f"{STORE_DIR}/index.faiss")
    with open(f"{STORE_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Index saved to '{STORE_DIR}/'.")


if __name__ == "__main__":
    main()
