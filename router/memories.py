from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from agent import store

router = APIRouter()

def _require_api_key(req: Request):
    want = os.getenv("ACTIONS_API_KEY")
    if not want: return
    got = req.headers.get("x-api-key")
    if got != want:
        raise HTTPException(status_code=401, detail="invalid api key")

class MemoryUpsert(BaseModel):
    type: str                        # semantic | episodic | procedural
    title: Optional[str] = None
    text: str
    tags: Optional[List[str]] = []
    role_view: Optional[List[str]] = []

@router.post("/memories/upsert")
async def memories_upsert(req: Request, body: MemoryUpsert):
    _require_api_key(req)
    mem_id = store.insert_memory(
        type=body.type, title=body.title or "", text=body.text,
        tags=body.tags or [], source="api", role_view=body.role_view or []
    )
    return {"memory_id": mem_id}
