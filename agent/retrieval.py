from vendors.openai_client import client, EMBED_MODEL
from vendors.pinecone_client import INDEX
from datetime import datetime, timezone

def embed(text: str):
    # keep it short for embeddings
    text = text[:8000]
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def upsert_memory_vector(mem_id: str, user_id: str | None, type_: str, content: str, title: str, tags, importance: int, created_at_iso: str | None = None):
    vec = embed(content)
    meta = {
        "memory_id": mem_id,
        "user_id": str(user_id) if user_id else None,
        "type": type_,
        "title": (title or "")[:120],
        "tags": tags or [],
        "importance": int(importance),
        "created_at": created_at_iso or datetime.now(timezone.utc).isoformat(),
        "deleted": False
    }
    INDEX.upsert([{"id": mem_id, "values": vec, "metadata": meta}])

def search(user_id: str | None, query: str, top_k=6, types=None):
    vec = embed(query)
    flt = {"deleted": False}
    if user_id: flt["user_id"] = str(user_id)
    if types:   flt["type"] = {"$in": types}
    res = INDEX.query(vector=vec, top_k=top_k, include_metadata=True, filter=flt)
    return res.matches or []
