from __future__ import annotations

import os, hashlib
from typing import Optional, List
from fastapi import APIRouter, Header, HTTPException, Body
from pydantic import BaseModel

from agent import store
from vendors.openai_client import client, EMBED_MODEL
from vendors.pinecone_client import get_index

router = APIRouter(tags=["memories"])

def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

class MemoriesUpsertBody(BaseModel):
    type: str  # 'episodic' | 'semantic' | 'procedural'
    title: Optional[str] = None
    text: str
    tags: Optional[List[str]] = None
    role_view: Optional[List[str]] = None

def _embed(text: str):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

@router.post("/memories/upsert")
def memories_upsert(
    body: MemoriesUpsertBody = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)

    # Normalize + dedupe hash
    norm_text = " ".join((body.text or "").split()).strip()
    if not norm_text:
        raise HTTPException(status_code=400, detail="text is required")
    dedupe_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

    # Upsert memory row (dedupe by hash)
    existing = store.find_memory_by_dedupe_hash(dedupe_hash)
    if existing:
        # Already exists; return what we have
        return {
            "memory_id": existing["id"],
            "embedding_id": existing.get("embedding_id"),
        }

    # Insert new memory
    mem_row = store.insert_memory(
        type=body.type,
        title=body.title or "",
        text=norm_text,
        tags=body.tags or [],
        source="api",
        role_view=body.role_view or [],
        dedupe_hash=dedupe_hash,
    )

    # Embed + Pinecone upsert
    try:
        vec = _embed(norm_text)
        emb_id = f"mem_{mem_row['id']}"
        idx = get_index()
        idx.upsert(
            vectors=[{
                "id": emb_id,
                "values": vec,
                "metadata": {
                    "type": body.type,
                    "title": body.title or "",
                    "tags": body.tags or [],
                    "source": "api",
                }
            }],
            namespace=body.type,  # semantic | episodic | procedural
        )
        store.update_memory_embedding_id(mem_row["id"], emb_id)
    except Exception as ex:
        # Don’t fail the request; just return without embedding
        emb_id = None

    return {"memory_id": mem_row["id"], "embedding_id": emb_id}
