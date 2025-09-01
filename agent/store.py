# agent/store.py
from __future__ import annotations

from typing import Optional, Dict, Any, List
from vendors.supabase_client import supabase


# ----------------------------
# Users
# ----------------------------
def upsert_user(email: str) -> Dict[str, Any]:
    """
    Idempotently create (or fetch) a user by email.
    Supabase-py v2: upsert + select to return the row.
    """
    supabase.table("users").upsert({"email": email}, on_conflict="email").execute()
    res = supabase.table("users").select("*").eq("email", email).limit(1).execute()
    if not res.data:
        raise RuntimeError("failed to upsert/select user by email")
    return res.data[0]


def ensure_user(user_id: Optional[str], user_email: Optional[str]) -> Dict[str, Any]:
    """
    Resolve a user by UUID or email (create-by-email if needed).
    """
    # UUID path
    if user_id:
        r = supabase.table("users").select("*").eq("id", user_id).limit(1).execute()
        if r.data:
            return r.data[0]
        # fall back to email creation if provided
        if user_email:
            return upsert_user(user_email)
        raise ValueError("user_id not found and no user_email provided")

    # Email path
    if user_email:
        return upsert_user(user_email)

    # Fallback (dev): anonymous
    return upsert_user("anonymous@suaps.local")


# ----------------------------
# Sessions
# ----------------------------
def create_session(user_id: str) -> Dict[str, Any]:
    """
    Create a session row for the user and return at least an id.
    """
    ins = supabase.table("sessions").insert({"user_id": user_id}).select("id").execute()
    if not ins.data:
        # Some stacks require a follow-up select
        sel = supabase.table("sessions").select("id").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if not sel.data:
            raise RuntimeError("failed to create/select session")
        return {"id": sel.data[0]["id"]}
    return {"id": ins.data[0]["id"]}


# ----------------------------
# Memories
# ----------------------------
def upsert_memory(
    user_id: str,
    type_: str,
    title: str,
    content: str,
    importance: int = 3,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Insert a memory row.

    IMPORTANT: Your DB has a NOT NULL constraint on `value`. To satisfy the schema
    while keeping the API's `content` field, we write BOTH `content` and `value`.
    """
    payload = {
        "user_id": user_id,
        "type": type_,
        "title": title or "",
        "content": content,           # API's field
        "value": content,             # satisfy DB constraint
        "importance": importance,
        "tags": tags or [],
    }

    res = supabase.table("memories").insert(payload).select("id").execute()
    if not res.data:
        # Older PostgREST setups sometimes return no data on insert without .select().execute()
        # Fallback to selecting the newest row for this user with the same title+content.
        sel = (
            supabase.table("memories")
            .select("id")
            .eq("user_id", user_id)
            .eq("title", payload["title"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not sel.data:
            raise RuntimeError("insert memory returned no row")
        return {"id": sel.data[0]["id"]}

    return {"id": res.data[0]["id"]}


# ----------------------------
# Optional: message logging (used by pipeline)
# ----------------------------
def log_message(session_id: str, role: str, content: str) -> None:
    try:
        supabase.table("messages").insert(
            {"session_id": session_id, "role": role, "content": content}
        ).execute()
    except Exception:
        # Don't break the request if logging fails
        pass
