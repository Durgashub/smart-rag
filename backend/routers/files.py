"""
routers/files.py — file management endpoints.
"""

import sys
import subprocess
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Header

from services.sessions import get_session_dirs, get_files_for_session, delete_index

router = APIRouter()


@router.get("/api/files")
def list_files(x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        return {"files": []}
    return {"files": get_files_for_session(x_session_id)}


@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(None),
):
    """
    Save file to docs/{session_id}/ then run ingest.py via subprocess.

    Uses sys.executable (not bare 'python') to stay inside the correct venv.
    """
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    docs_dir, _ = get_session_dirs(x_session_id)
    destination = docs_dir / file.filename
    with open(destination, "wb") as f:
        f.write(await file.read())
    subprocess.run([sys.executable, "ingest.py", "--session", x_session_id], check=True)
    return {"files": get_files_for_session(x_session_id)}


@router.delete("/api/files/{filename}")
def delete_file(
    filename: str,
    x_session_id: Optional[str] = Header(None),
):
    """Delete file and re-index. If no files remain, delete the index."""
    if not x_session_id:
        raise HTTPException(status_code=401, detail="No session ID.")
    docs_dir, _ = get_session_dirs(x_session_id)
    file_path   = docs_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    file_path.unlink()
    remaining = get_files_for_session(x_session_id)
    if remaining:
        subprocess.run([sys.executable, "ingest.py", "--session", x_session_id], check=True)
    else:
        delete_index(x_session_id)
    return {"message": f"{filename} deleted.", "files": remaining}
