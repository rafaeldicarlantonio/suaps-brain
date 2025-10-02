# router/memories.py
import os
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.pipeline import upsert_memories_from_chunks, normalize_text
from auth.light_identity import ensure_user  # attribution

router = APIRouter()

class UpsertReq(BaseModel):
    type: str = Field(..., pattern="^(semantic|episodic|procedural)$")
    title: Optional[str] = None
    text: str
    tags: Optional[List[str]] = None
    role_view: Optional[List[str]] = None
    source: Optional[str] = "chat"
    file_id: Optional[str] = None

class UpsertResp(BaseModel):
    memory_id: Optional[str] = None
    embedding_id: Optional[str] = None

def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.post("/memories/upsert", response_model=UpsertResp)
def memories_upsert_post(
    body: UpsertReq,
    x_api_key: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None),
):
    _auth(x_api_key)
    sb = get_client()
    index = get_index()
    author_user_id = ensure_user(sb=sb, email=x_user_email)

    # Use pipeline to reuse dedupe/embedding logic
    res = upsert_memories_from_chunks(
        sb=sb,
        pinecone_index=index,
        embedder=None,
        file_id=body.file_id,
        title_prefix=body.title or "Note",
        chunks=[normalize_text(body.text)],
        mem_type=body.type,
        tags=body.tags or [],
        role_view=body.role_view or [],
        source=body.source or "chat",
        text_col_env=os.getenv("MEMORIES_TEXT_COLUMN", "text"),
        author_user_id=author_user_id,  # <-- colon in dicts is correct; this is a keyword arg here
    )

    # prefer newly created id if present; else updated id if any
    mem_id = None
    if res.get("upserted"):
        mem_id = res["upserted"][0].get("memory_id")
    elif res.get("updated"):
        mem_id = res["updated"][0].get("memory_id")

    return {"memory_id": mem_id, "embedding_id": f"mem_{mem_id}" if mem_id else None}
