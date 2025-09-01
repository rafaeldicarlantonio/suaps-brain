# agent/store.py
# Supabase-backed store helpers used by the app and autosave

from __future__ import annotations
from typing import Optional, Dict, Any, List
from hashlib import sha256

from vendors.supabase_client import supabase  # expects a configured client

__all__ = [
    "supabase",
    "ensure_user",
    "create_session",
    "ensure_session",
    "insert_message",
    "upsert_memory",
    "insert_memory",
    "find_memory_by_dedupe_hash",
    "update_memory_embedding_id",
    "log_tool_run",
]

# -------------------------
# Users / Sessions / Messages
# -------------------------
def ensure_user(user_id: Optional[str], email: Optional[str]) -> Dict[str, Any]:
    email = (email or "anonymous@suaps.local").strip().lower()
    if user_id:
        r = supabase.table("users").select("id,email").eq("id", user_id).limit(1).execute()
        if r.data:
            return r.data[0]
    r = supabase.table("users").select("id,email").eq("email", email).limit(1).execute()
    if r.data:
        return r.data[0]
    ins = supabase.table("users").insert({"email": email}).execute()
    return ins.data[0]

def create_session(user_id: str) -> Dict[str, Any]:
    r = supabase.table("sessions").insert({"user_id": user_id}).execute()
    return r.data[0]

def ensure_session(session_id: Optional[str], user_id: str) -> Dict[str, Any]:
    if session_id:
        r = supabase.table("sessions").select("id,user_id").eq("id", session_id).limit(1).execute()
        if r.data:
            return r.data[0]
    return create_session(user_id)

def insert_message(session_id: str, role: str, content: str, model: Optional[str]=None, tokens: Optional[int]=None, latency_ms: Optional[int]=None) -> Dict[str, Any]:
    row = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "model": model,
        "tokens": tokens,
        "latency_ms": latency_ms,
    }
    r = supabase.table("messages").insert(row).execute()
    return r.data[0]

# -------------------------
# Memories
# -------------------------
def _dedupe_hash(text: str) -> str:
    norm = " ".join((text or "").split()).strip()
    return sha256(norm.encode("utf-8")).hexdigest()

def find_memory_by_dedupe_hash(dedupe_hash: str) -> Optional[Dict[str, Any]]:
    r = supabase.table("memories").select("*").eq("dedupe_hash", dedupe_hash).limit(1).execute()
    return r.data[0] if r.data else None

def insert_memory(*, type: str, title: str, text: str, tags: List[str], source: str, file_id: Optional[str], session_id: Optional[str], author_user_id: Optional[str], role_view: Optional[List[str]], dedupe_hash: Optional[str]) -> str:
    row = {
        "type": type,
        "title": title or "",
        "text": text,
        "tags": tags or [],
        "source": source,
        "file_id": file_id,
        "session_id": session_id,
        "author_user_id": author_user_id,
        "role_view": role_view or [],
        "dedupe_hash": dedupe_hash,
    }
    r = supabase.table("memories").insert(row).execute()
    return r.data[0]["id"]

def update_memory_embedding_id(memory_id: str, embedding_id: str) -> None:
    supabase.table("memories").update({"embedding_id": embedding_id}).eq("id", memory_id).execute()

def upsert_memory(user_id: str, type_: str, title: str, content: str, importance: int, tags: List[str]) -> Dict[str, Any]:
    dh = _dedupe_hash(content)
    existing = find_memory_by_dedupe_hash(dh)
    if existing:
        return existing
    row = {
        "type": type_,
        "title": title or "",
        "text": content,
        "tags": tags or [],
        "source": "chat",
        "file_id": None,
        "session_id": None,
        "author_user_id": user_id,
        "role_view": [],
        "dedupe_hash": dh,
    }
    ins = supabase.table("memories").insert(row).execute()
    return ins.data[0]

# -------------------------
# Tool runs / logging
# -------------------------
def log_tool_run(name: str, input_json, output_json, success: bool, latency_ms: int | None) -> None:
    row = {
        "name": name,
        "input_json": input_json,
        "output_json": output_json,
        "success": bool(success),
        "latency_ms": int(latency_ms or 0),
    }
    try:
        supabase.table("tool_runs").insert(row).execute()
    except Exception:
        # never crash the main flow due to logging
        pass
