# router/upload.py
import os
from typing import Optional, List
from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form
from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.converters import sniff_and_convert
from ingest.pipeline import normalize_text, chunk_text, upsert_memories_from_chunks

router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),             # csv or leave empty
    type: Optional[str] = Form("semantic"),       # semantic default
    x_api_key: Optional[str] = Header(None),
):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if type not in ("semantic","episodic","procedural"):
        raise HTTPException(status_code=400, detail="type must be one of semantic|episodic|procedural")

    raw = await file.read()
    text, mime = sniff_and_convert(file.filename, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="no text extracted from file")

    sb = get_client()
    # save file row
    frow = {
        "filename": file.filename,
        "mime_type": mime,
        "bytes": len(raw),
        "storage_url": "",
        "text_extracted": text,
    }
    ins = sb.table("files").insert(frow).execute()
    # grab id
    sel = sb.table("files").select("id").order("created_at", desc=True).limit(1).execute()
    fid = (sel.data or [{}])[0].get("id")

    # chunk and ingest
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","2000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP","200"))
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    pinecone_index = get_index()
    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    res = upsert_memories_from_chunks(
        sb=sb,
        pinecone_index=pinecone_index,
        embedder=None,                    # internal in pipeline
        file_id=fid,
        title_prefix=file.filename,
        chunks=chunks,
        mem_type=type,
        tags=tags_list,
        role_view=[],
        source="upload",
        text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","value"),
    )

    return {
        "file_id": fid,
        "bytes": len(raw),
        "mime_type": mime,
        "chunks": len(chunks),
        "ingest_result": res,
    }
