from vendors.supabase_client import supabase
from typing import Optional
import re

def is_uuid(val: str) -> bool:
    return bool(re.match(r"^[0-9a-fA-F-]{36}$", val or ""))

def upsert_user(email: str) -> dict:
    # upsert by unique email
    res = supabase.table("users").upsert({"email": email}, on_conflict="email").select("*").execute()
    return res.data[0]

def get_user(user_id: str) -> Optional[dict]:
    if not is_uuid(user_id):
        return None
    res = supabase.table("users").select("*").eq("id", user_id).limit(1).execute()
    return res.data[0] if res.data else None

def ensure_user(user_id: Optional[str], user_email: Optional[str]) -> dict:
    """
    Returns a users row. If user_id provided but not found and email is present,
    will upsert by email. If only email is provided, upserts and returns.
    """
    if user_id:
        row = get_user(user_id)
        if row:
            return row
        # if ID invalid/missing but we have email, fall through to upsert by email
    if not user_email:
        raise ValueError("user_email required when user_id is missing or invalid")
    return upsert_user(user_email)

def create_session(user_id: str, channel="gpt_actions"):
    return supabase.table("sessions").insert({"user_id": user_id, "channel": channel}).execute().data[0]

def add_message(session_id, user_id, role, content, tokens=None):
    supabase.table("messages").insert({"session_id": session_id, "user_id": user_id, "role": role, "content": content, "tokens": tokens}).execute()

def upsert_memory(user_id, type_, title, content, importance=3, tags=None, created_at=None):
    row = {"user_id": user_id, "type": type_, "title": title, "content": content, "importance": importance, "tags": tags or []}
    if created_at:
        row["created_at"] = created_at
    data = supabase.table("memories").insert(row).execute().data[0]
    return data
