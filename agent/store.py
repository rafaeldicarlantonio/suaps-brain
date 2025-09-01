# agent/store.py
# PRD-aligned Supabase CRUD helpers for memories/entities/tool_runs
from __future__ import annotations
import json, time, hashlib
from typing import Any, Dict, List, Optional, Tuple
from vendors.supabase_client import supabase

def _exec(q):
    try:
        r = q.execute()
        return r.data or []
    except Exception as ex:
        raise RuntimeError(f"Supabase error: {ex}")

# ---- tool_runs (observability) ----
def log_tool_run(name: str, input_json: Dict[str, Any], output_json: Dict[str, Any], success: bool, latency_ms: int) -> None:
    row = dict(name=name, input_json=input_json, output_json=output_json, success=bool(success), latency_ms=int(latency_ms))
    try:
        _exec(supabase.table("tool_runs").insert(row))
    except Exception:
        # best-effort logging
        pass

# ---- memories ----
def sha256_normalized(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def find_memory_by_dedupe_hash(dedupe_hash: str) -> Optional[Dict[str, Any]]:
    rows = _exec(supabase.table("memories").select("*").eq("dedupe_hash", dedupe_hash).limit(1))
    return rows[0] if rows else None

def fetch_recent_memories_texts(limit: int = 500) -> List[Dict[str, Any]]:
    rows = _exec(supabase.table("memories").select("id,text,content,created_at").order("created_at", desc=True).limit(limit))
    # compat: prefer text over content
    for r in rows:
        r["text"] = r.get("text") or r.get("content") or ""
    return rows

def insert_memory(*, type: str, title: str, text: str, tags: List[str] | None = None,
                  source: Optional[str] = None, file_id: Optional[str] = None,
                  session_id: Optional[str] = None, author_user_id: Optional[str] = None,
                  role_view: Optional[List[str]] = None, dedupe_hash: Optional[str] = None) -> str:
    row = {
        "type": type,
        "title": title,
        "text": text,
        "content": text,  # backward compatibility
        "tags": tags or [],
        "source": source,
        "file_id": file_id,
        "session_id": session_id,
        "author_user_id": author_user_id,
        "role_view": role_view or None,
        "dedupe_hash": dedupe_hash,
    }
    rows = _exec(supabase.table("memories").insert(row).select("id").limit(1))
    return rows[0]["id"]

def update_memory_embedding_id(memory_id: str, embedding_id: str) -> None:
    _exec(supabase.table("memories").update({"embedding_id": embedding_id}).eq("id", memory_id))

# ---- entities & graph ----
def get_or_create_entity(name: str, type: str) -> str:
    # try find
    rows = _exec(supabase.table("entities").select("id").eq("name", name).eq("type", type).limit(1))
    if rows:
        return rows[0]["id"]
    # create
    rows = _exec(supabase.table("entities").insert({"name": name, "type": type}).select("id").limit(1))
    return rows[0]["id"]

def link_entity_to_memory(entity_id: str, memory_id: str, weight: float = 1.0) -> None:
    _exec(supabase.table("entity_mentions").upsert({"entity_id": entity_id, "memory_id": memory_id, "weight": weight}))

def add_entity_edge(src_id: str, dst_id: str, rel: str, weight: float = 1.0) -> None:
    _exec(supabase.table("entity_edges").upsert({"src": src_id, "dst": dst_id, "rel": rel, "weight": weight}))
