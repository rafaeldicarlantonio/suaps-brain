# router/ingest.py
import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.pipeline import normalize_text, chunk_text, upsert_memories_from_chunks

router = APIRouter()

class IngestItem(BaseModel):
    title: str
    text: str
    type: str = Field("semantic", regex="^(semantic|episodic|procedural)$")
    tags: List[str] = []
    source: str = "ingest"
    role_view: List[str] = []
    file_id: Optional[str] = None

class IngestBatch(BaseModel):
    items: List[IngestItem]
    dedupe: bool = True  # future use; exact dedupe is always on

@router.post("/ingest/batch")
def ingest_batch(body: IngestBatch, x_api_key: Optional[str] = Header(None)):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    sb = get_client()
    index = get_index()
    results = {"upserted": [], "skipped": []}

    for it in body.items:
        text = normalize_text(it.text)
        if not text:
            results["skipped"].append({"title": it.title, "reason": "empty"})
            continue
       CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP","200"))
if len(text) > CHUNK_SIZE:
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
else:
    chunks = [text]  # callers provide pre-chunked text; single chunk here
        r = upsert_memories_from_chunks(
            sb=sb,
            pinecone_index=index,
            embedder=None,
            file_id=it.file_id,
            title_prefix=it.title,
            chunks=chunks,
            mem_type=it.type,
            tags=it.tags,
            role_view=it.role_view,
            source=it.source,
            text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","value"),
        )
        results["upserted"].extend(r.get("upserted", []))
        results["skipped"].extend(r.get("skipped", []))

    return results
