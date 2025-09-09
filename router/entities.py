# router/entities.py
import os
from typing import Optional, List, Literal
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from vendors.supabase_client import get_client

router = APIRouter()

def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

class EntityItem(BaseModel):
    id: str
    name: str
    type: Literal["person","org","project","artifact","concept"]

@router.get("/entities/search")
def entities_search(
    q: str = Query(..., min_length=2),
    type: Optional[Literal["person","org","project","artifact","concept"]] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    x_api_key: Optional[str] = Header(None),
):
    _auth(x_api_key)
    sb = get_client()
    qry = sb.table("entities").select("id,name,type").ilike("name", f"%{q}%").limit(limit)
    if type:
        qry = qry.eq("type", type)
    res = qry.execute()
    data = res.data if hasattr(res, "data") else res.get("data") or []
    return {"items": data}

@router.get("/entities/{entity_id}/memories")
def entity_memories(
    entity_id: str,
    limit: int = Query(20, ge=1, le=100),
    x_api_key: Optional[str] = Header(None),
):
    _auth(x_api_key)
    sb = get_client()
    # join mentions -> memories
    m = sb.table("entity_mentions").select("memory_id").eq("entity_id", entity_id).limit(200).execute()
    mem_ids = [r["memory_id"] for r in (m.data if hasattr(m,"data") else m.get("data") or [])]
    if not mem_ids:
        return {"items": []}
    rows = sb.table("memories").select("id,type,title,tags,created_at").in_("id", mem_ids).order("created_at", desc=True).limit(limit).execute()
    data = rows.data if hasattr(rows,"data") else rows.get("data") or []
    return {"items": data}

@router.get("/entities/{entity_id}/neighbors")
def entity_neighbors(
    entity_id: str,
    x_api_key: Optional[str] = Header(None),
):
    _auth(x_api_key)
    sb = get_client()
    edges = sb.table("entity_edges").select("src,dst,rel,weight").or_(f"src.eq.{entity_id},dst.eq.{entity_id}").limit(200).execute()
    data = edges.data if hasattr(edges,"data") else edges.get("data") or []
    return {"items": data}
