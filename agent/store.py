from __future__ import annotations
from typing import Optional, Dict, Any, List
from hashlib import sha256
from vendors.supabase_client import supabase
from postgrest.exceptions import APIError

def ensure_session(session_id: Optional[str], title: Optional[str]) -> Dict[str,Any]:
    if session_id:
        # verify session exists
        r = supabase.table("sessions").select("*").eq("id", session_id).limit(1).execute()
        if (r.data):
            return r.data[0]
    # create
    r = supabase.table("sessions").insert({"title": title}).execute()
    return r.data[0]

def fetch_recent_messages(session_id: str, limit: int = 4) -> List[Dict[str,Any]]:
    r = supabase.table("messages").select("*").eq("session_id", session_id).order("created_at", desc=True).limit(limit).execute()
    return list(reversed(r.data or []))

def insert_message(session_id: str, role: str, content: str, model: Optional[str], tokens: Optional[int], latency_ms: Optional[int]):
    supabase.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "model": model,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }).execute()

def find_memory_by_dedupe_hash(dh: str) -> Optional[Dict[str,Any]]:
    r = supabase.table("memories").select("*").eq("dedupe_hash", dh).limit(1).execute()
    return (r.data or [None])[0]

def insert_memory(mem: Optional[Dict[str,Any]] = None, **kwargs) -> Dict[str,Any]:
    """
    Accept dict or kwargs; if DB enforces user_id NOT NULL, retry with DEFAULT_USER_ID.
    """
    if mem is None:
        mem = {}
    if kwargs:
        mem.update(kwargs)

    try:
        r = supabase.table("memories").insert(mem).execute()
        return r.data[0]
    except APIError as ex:
        msg = (getattr(ex, "message", "") or "").lower()
        if "user_id" in msg and "not-null" in msg or "not null" in msg:
            default_user = os.getenv("SUPABASE_DEFAULT_USER_ID")
            if not default_user:
                raise  # no fallback available
            mem2 = dict(mem)
            mem2["user_id"] = default_user
            r = supabase.table("memories").insert(mem2).execute()
            return r.data[0]
        raise


def upsert_memory(mem: Dict[str,Any]) -> Dict[str,Any]:
    dh = mem.get("dedupe_hash")
    if dh:
        ex = find_memory_by_dedupe_hash(dh)
        if ex:
            return ex
    return insert_memory(mem)

def update_memory_embedding_id(memory_id: str, embedding_id: str):
    supabase.table("memories").update({"embedding_id": embedding_id}).eq("id", memory_id).execute()

def log_tool_run(name: str, input_json: Any, output_json: Any, success: bool, latency_ms: Optional[int] = None):
    supabase.table("tool_runs").insert({
        "name": name, "input_json": input_json, "output_json": output_json, "success": success, "latency_ms": latency_ms
    }).execute()
