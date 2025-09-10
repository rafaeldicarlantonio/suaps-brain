# auth/light_identity.py
from typing import Optional
from vendors.supabase_client import get_client

def ensure_user(sb=None, email: Optional[str]=None, name: Optional[str]=None, role: Optional[str]=None) -> Optional[str]:
    """
    Idempotent: returns users.id for the given email, creating a row if needed.
    If the users table or column doesn't exist, returns None (no crash).
    """
    if not email:
        return None
    try:
        sb = sb or get_client()
        q = sb.table("users").select("id").eq("email", email).limit(1).execute()
        rows = q.data if hasattr(q, "data") else q.get("data") or []
        if rows:
            return rows[0]["id"]
        payload = {"email": email}
        if name: payload["name"] = name
        if role: payload["role"] = role
        sb.table("users").insert(payload).execute()
        q2 = sb.table("users").select("id").eq("email", email).limit(1).execute()
        rows2 = q2.data if hasattr(q2, "data") else q2.get("data") or []
        return rows2[0]["id"] if rows2 else None
    except Exception:
        # Table may not exist yet — attribution is best-effort
        return None
