from __future__ import annotations
import os, io, time, mimetypes
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, Header, HTTPException

from vendors.supabase_client import supabase
from agent.ingest import clean_text, ingest_text

router = APIRouter(tags=["upload"])

def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)
    raw = await file.read()
    mime = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    text = clean_text(raw, mime, filename=file.filename)
    # persist file row
    r = supabase.table("files").insert({
        "filename": file.filename,
        "mime_type": mime,
        "bytes": len(raw),
        "text_extracted": text,
        "storage_url": None,
    }).execute()
    file_row = r.data[0]
    # ingest normalized text
    res = ingest_text(text=text, title=file.filename, type="semantic", tags=(tags or "").split(",") if tags else [], source="upload", role_view=None, file_id=file_row["id"])
    return {"file_id": file_row["id"], "bytes": len(raw), "mime_type": mime, "ingest_job_id": res.get("upserted",[])[0].get("embedding_id") if res.get("upserted") else None}
