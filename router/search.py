# router/search.py
import os
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index, safe_query

router = APIRouter()
client = OpenAI()

class SearchReq(BaseModel):
    q: str
    type: Optional[List[str]] = Field(default_factory=lambda: ["semantic", "episodic", "procedural"])
    top_k: int = Field(12, ge=1, le=50)
    include_text: bool = True

class SearchItem(BaseModel):
    id: str
    type: str
    title: Optional[str] = None
    score: float
    text: Optional[str] = None

class SearchResp(BaseModel):
    items: List[SearchItem]

def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _embed(text: str) -> List[float]:
    kwargs: Dict[str, Any] = {"model": os.getenv("EMBED_MODEL", "text-embedding-3-small"), "input": text}
    dim = os.getenv("EMBED_DIM")
    if dim:
        kwargs["dimensions"] = int(dim)
    er = client.embeddings.create(**kwargs)
    return er.data[0].embedding

@router.post("/search/semantic", response_model=SearchResp)
def search_semantic_post(body: SearchReq, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    sb = get_client()
    index = get_index()

    qvec = _embed(body.q)
    types = body.type or ["semantic", "episodic", "procedural"]

    matches: List[Dict[str, Any]] = []
    for t in types:
        res = safe_query(index, vector=qvec, top_k=body.top_k, include_metadata=True, namespace=t)
        for m in (res.matches or []):
            md = m.metadata or {}
            mem_id = (md.get("id") or (m.id or "")).replace("mem_", "")
            if not mem_id:
                continue
            matches.append({"memory_id": mem_id, "type": t, "score": float(m.score or 0.0)})

    ids = list({m["memory_id"] for m in matches})
    text_col = (os.getenv("MEMORIES_TEXT_COLUMN", "text")).strip().lower()
    by_id: Dict[str, Dict[str, Any]] = {}

    if ids:
        sel_cols = f"id,type,title,{text_col}" if body.include_text else "id,type,title"
        rows = sb.table("memories").select(sel_cols).in_("id", ids).limit(len(ids)).execute()
        data = rows.data if hasattr(rows, "data") else rows.get("data") or []
        by_id = {r["id"]: r for r in data}

    out: List[Dict[str, Any]] = []
    for m in matches:
        r = by_id.get(m["memory_id"])
        if not r:
            continue
        out.append(
            {
                "id": r["id"],
                "type": r["type"],
                "title": r.get("title"),
                "score": m["score"],
                "text": r.get(text_col) if body.include_text else None,
            }
        )

    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return {"items": out[: body.top_k]}
