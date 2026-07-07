"""
routers/system.py — system endpoints: health, home, session.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Header

router = APIRouter()


@router.get("/api/health")
def health():
    return {"status": "ok", "stage": 4}


@router.get("/")
def home():
    return {"message": "SmartRAG AI Backend — Stage 4 (Modular)"}


@router.get("/api/session")
def create_session(x_session_id: Optional[str] = Header(None)):
    """Return existing session ID or create a new UUID."""
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
    return {"session_id": x_session_id}
