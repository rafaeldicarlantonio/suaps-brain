from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from agent.ingest import ingest_text

router = APIRouter()

def _require_api_key(req: Request):
    want = os.getenv("ACTIONS_API_KEY")
    if not want: return
    got = req.headers.get("x-api-key")
    if got != want:
        raise HTTPException(status_code=401, detail="invalid api key")

class IngestItem(BaseModel):
    text: str
    type: str  # "semantic" | "episodic" | "procedural"
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    role_view: Optional[List[str]] = None

class IngestBatchRequest(BaseModel):
    items: List[IngestItem]

@router.post("/ingest/batch")
async def ingest_batch(req: Request, body: IngestBatchRequest):
    _require_api_key(req)
    upserted, skipped = [], []
    for it in body.items:
        try:
            mid, emb = ingest_text(
                text=it.text, _type=it.type, title=it.title or "",
                tags=it.tags or [], source=it.source or "api",
                role_view=it.role_view or []
            )
            upserted.append({"memory_id": mid, "embedding_id": emb})
        except Exception as ex:
            skipped.append({"reason": str(ex)})
    return {"upserted": upserted, "skipped": skipped}
