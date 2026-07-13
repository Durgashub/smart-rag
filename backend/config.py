"""
config.py — ALL constants and the shared OpenAI client.

Every other module imports from here.
To tune the system, change values in this ONE file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the same directory as this file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── OpenAI client (shared singleton) ─────────────────────────────────────────
client = OpenAI()

# ── Models ────────────────────────────────────────────────────────────────────
CHAT_MODEL  = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"

# ── Conversation memory ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 10

# ── Chunking ──────────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE      = 1200
CHILD_CHUNK_SIZE       = 400
CHILD_OVERLAP          = 50
CHUNK_OVERLAP          = 200
MIN_TEXT_LEN_BEFORE_OCR = 20

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 6
RRF_K         = 60

# ── Cross-encoder ─────────────────────────────────────────────────────────────
CROSS_ENCODER_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_NOISE_FLOOR = -5.0

# ── Accuracy display ──────────────────────────────────────────────────────────
FAISS_MIN_DIST = 0.3
FAISS_MAX_DIST = 1.5

# ── OCR ───────────────────────────────────────────────────────────────────────
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
POPPLER_PATH  = os.getenv("POPPLER_PATH")

# ── CORS ──────────────────────────────────────────────────────────────────────
# IMPORTANT: update this list every time the frontend URL changes
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://chat-smart.lovable.app",
    "https://dociq.up.railway.app",
]
ALLOWED_ORIGIN_REGEX = r"https://.*\.lovable\.app|https://.*\.lovableproject\.com|https://.*\.up\.railway\.app"

# ── Supported file types ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls",
    ".csv", ".txt", ".md", ".pptx",
}