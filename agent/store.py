from __future__ import annotations
from typing import Optional, Dict, Any, List
from hashlib import sha256
from vendors.supabase_client import supabase

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
    Accepts either a dict or keyword args (for compatibility with older callers).
    Example:
        insert_memory({"type":"semantic", ...})
        insert_memory(type="semantic", title="...", text="...", tags=[], source="api", role_view=[])
    """
    if mem is None:
        mem = {}
    if kwargs:
        mem.update(kwargs)
    r = supabase.table("memories").insert(mem).execute()
    return r.data[0]


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
