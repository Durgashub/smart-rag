"""
config.py — ALL constants and the shared OpenAI client.

Every other module imports from here.
To tune the system, change values in this ONE file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same directory as this file — works regardless of
# which directory uvicorn is launched from
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── OpenAI client (shared singleton) ─────────────────────────────────────────
client = OpenAI()

# ── Models ────────────────────────────────────────────────────────────────────
CHAT_MODEL  = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

# ── Conversation memory ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10          # last N Q&A pairs kept in context

# ── Chunking ──────────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE      = 1200   # sent to GPT for richer answers
CHILD_CHUNK_SIZE       = 400    # embedded for retrieval + cross-encoder
CHILD_OVERLAP          = 50     # overlap between child chunks
CHUNK_OVERLAP          = 200    # overlap between parent chunks
MIN_TEXT_LEN_BEFORE_OCR = 20   # pages below this char count trigger OCR

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 6
RRF_K         = 60              # Reciprocal Rank Fusion constant

# ── Cross-encoder ─────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_NOISE_FLOOR = -5.0  # scores below → fall back to FAISS distance

# ── Accuracy display ──────────────────────────────────────────────────────────
FAISS_MIN_DIST = 0.3            # → 92% accuracy
FAISS_MAX_DIST = 1.5            # → 46% accuracy

# ── OCR ───────────────────────────────────────────────────────────────────────
TESSERACT_CMD = os.getenv("TESSERACT_CMD")   # Windows path override
POPPLER_PATH  = os.getenv("POPPLER_PATH")    # Windows path override

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://chat-smart.lovable.app",
]
ALLOWED_ORIGIN_REGEX = r"https://.*\.lovable\.app|https://.*\.lovableproject\.com"

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".csv", ".txt", ".md", ".pptx",
}
