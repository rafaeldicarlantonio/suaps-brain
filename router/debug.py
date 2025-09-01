# router/debug.py
import os
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from vendors.supabase_client import supabase

router = APIRouter()

def auth(x_api_key: Optional[str]):
    want = os.getenv("ACTIONS_API_KEY") or "dev_key"
    if (os.getenv("DISABLE_AUTH","false").lower() == "true"):
        return
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="invalid X-API-Key")

@router.get("/debug/memories")
def debug_memories(x_api_key: Optional[str] = Header(None), limit: int = 20, type: Optional[str] = None):
    auth(x_api_key)
    q = supabase.table("memories").select("*").order("created_at", desc=True).limit(limit)
    if type:
        q = q.eq("type", type)
    r = q.execute()
    out = [{"id": row["id"], "type": row.get("type"), "title": row.get("title"), "tags": row.get("tags"), "created_at": row.get("created_at")} for row in (r.data or [])]
    return {"items": out}
