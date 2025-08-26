from vendors.supabase_client import supabase

def create_session(user_id: str, channel="gpt_actions"):
    return supabase.table("sessions").insert({"user_id": user_id, "channel": channel}).execute().data[0]

def add_message(session_id, user_id, role, content, tokens=None):
    supabase.table("messages").insert({"session_id": session_id, "user_id": user_id, "role": role, "content": content, "tokens": tokens}).execute()

def upsert_memory(user_id, type_, title, content, importance=3, tags=None, created_at=None):
    row = {"user_id": user_id, "type": type_, "title": title, "content": content, "importance": importance, "tags": tags or []}
    if created_at:
        row["created_at"] = created_at
    data = supabase.table("memories").insert(row).execute().data[0]
    return data  # includes id
