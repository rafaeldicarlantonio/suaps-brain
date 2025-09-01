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
    Supabase-py v2: .upsert(...).execute(); then .select(...) to fetch.
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

    NOTE: In supabase-py v2 you cannot chain .select() after .insert().
    """
    ins = supabase.table("sessions").insert({"user_id": user_id}).execute()
    # Many projects configure PostgREST to return the inserted row; if not, fetch it.
    if ins.data and isinstance(ins.data, list) and ins.data:
        row = ins.data[0]
        # If server didn't include id, fallback to select
        if "id" in row and row["id"]:
            return {"id": row["id"]}
    sel = (
        supabase.table("sessions")
        .select("id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not sel.data:
        raise RuntimeError("failed to create/select session")
    return {"id": sel.data[0]["id"]}


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

    IMPORTANT: DB has NOT NULL on `value`. We write BOTH `content` and `value`.
    Supabase v2: call .insert(...).execute(); don't chain .select().
    """
    payload = {
        "user_id": user_id,
        "type": type_,
        "title": title or "",
        "content": content,   # API field
        "value": content,     # satisfy DB constraint
        "importance": importance,
        "tags": tags or [],
    }

    ins = supabase.table("memories").insert(payload).execute()
    if ins.data and isinstance(ins.data, list) and ins.data:
        row = ins.data[0]
        if "id" in row and row["id"]:
            return {"id": row["id"]}

    # Fallback select by user + latest created (avoid strict content match if text is large)
    sel = (
        supabase.table("memories")
        .select("id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not sel.data:
        raise RuntimeError("insert memory returned no row")
    return {"id": sel.data[0]["id"]}


# ----------------------------
# Optional: message logging (used by pipeline)
# ----------------------------
def log_message(session_id: str, role: str, content: str) -> None:
    try:
        supabase.table("messages").insert(
            {"session_id": session_id, "role": role, "content": content}
        ).execute()
    except Exception:
        # Non-fatal
        pass
