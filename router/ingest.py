# router/ingest.py
import os
from typing import Optional, List, Dict, Any, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.pipeline import normalize_text, chunk_text, upsert_memories_from_chunks

router = APIRouter()


class IngestItem(BaseModel):
    title: str
    text: str
    # Use Literal for validation instead of regex= (removed in Pydantic v2)
    type: Literal["semantic", "episodic", "procedural"] = "semantic"
    tags: List[str] = Field(default_factory=list)
    source: str = "ingest"
    role_view: List[str] = Field(default_factory=list)
    file_id: Optional[str] = None


class IngestBatch(BaseModel):
    items: List[IngestItem]
    dedupe: bool = True  # reserved; exact dedupe enforced server-side


@router.post("/ingest/batch")
def ingest_batch(body: IngestBatch, x_api_key: Optional[str] = Header(None)):
    # Optional API key check
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    sb = get_client()
    index = get_index()

    results: Dict[str, List[Dict[str, Any]]] = {"upserted": [], "skipped": []}

    # Auto-chunking controls
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    for it in body.items:
        text = normalize_text(it.text or "")
        if not text:
            results["skipped"].append({"title": it.title, "reason": "empty"})
            continue

        # Auto-chunk long text; keep short text as single chunk
        if len(text) > CHUNK_SIZE:
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            chunks = [text]

        r = upsert_memories_from_chunks(
            sb=sb,
            pinecone_index=index,
            embedder=None,  # handled inside pipeline
            file_id=it.file_id,
            title_prefix=it.title,
            chunks=chunks,
            mem_type=it.type,
            tags=it.tags,
            role_view=it.role_view,
            source=it.source,
            text_col_env=os.getenv("MEMORIES_TEXT_COLUMN", "value"),
        )

        results["upserted"].extend(r.get("upserted", []))
        results["skipped"].extend(r.get("skipped", []))

    return results
