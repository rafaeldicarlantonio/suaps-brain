# router/ingest.py
import os
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.pipeline import upsert_memories_from_chunks, normalize_text
from auth.light_identity import ensure_user

router = APIRouter()

class IngestItem(BaseModel):
    title: Optional[str] = None
    text: str = Field(..., description="Normalized plain text")
    type: str = Field("semantic", pattern="^(semantic|episodic|procedural)$")
    tags: Optional[List[str]] = None
    source: Optional[str] = "upload"
    role_view: Optional[List[str]] = None
    file_id: Optional[str] = None

class IngestBatchRequest(BaseModel):
    items: List[IngestItem]
    dedupe: bool = True

class IngestBatchResponse(BaseModel):
    upserted: List[Dict[str, Any]]
    updated: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]

def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.post("/ingest/batch", response_model=IngestBatchResponse)
def ingest_batch_ingest_batch_post(body: IngestBatchRequest, x_api_key: Optional[str] = Header(None), x_user_email: Optional[str] = Header(None),):
    _auth(x_api_key)
    sb = get_client()
    author_user_id = ensure_user(sb=sb, email=x_user_email)
    index = get_index()

    # simple size guard to avoid huge single calls
    MAX_ITEMS = int(os.getenv("INGEST_MAX_ITEMS", "50"))
    if len(body.items) > MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Too many items; max {MAX_ITEMS}")

    by_type: Dict[str, List[str]] = {"semantic": [], "episodic": [], "procedural": []}
    titles_by_type: Dict[str, List[str]] = {"semantic": [], "episodic": [], "procedural": []}
    tags, role_view, source, file_id = [], [], "upload", None

    for it in body.items:
        t = normalize_text(it.text)
        if not t:
            continue
        by_type[it.type].append(t)
        titles_by_type[it.type].append(it.title or "Untitled")
        tags = it.tags or tags
        role_view = it.role_view or role_view
        source = it.source or source
        file_id = it.file_id or file_id

    all_upserted: List[Dict[str, Any]] = []
    all_updated: List[Dict[str, Any]] = []
    all_skipped: List[Dict[str, Any]] = []

    for ttype, texts in by_type.items():
        if not texts:
            continue
        resp = upsert_memories_from_chunks(
            sb=sb,
            pinecone_index=index,
            embedder=None,
            file_id=file_id,
            title_prefix=", ".join(titles_by_type[ttype][:2])[:80] or "Batch",
            chunks=texts,
            mem_type=ttype,
            tags=tags,
            role_view=role_view,
            source=source,
            text_col_env=os.getenv("MEMORIES_TEXT_COLUMN", "text"),
            author_user_id=author_user_id,
        )
        all_upserted.extend(resp.get("upserted", []))
        all_updated.extend(resp.get("updated", []))
        all_skipped.extend(resp.get("skipped", []))

    return {"upserted": all_upserted, "updated": all_updated, "skipped": all_skipped}
