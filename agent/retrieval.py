from vendors.openai_client import client, EMBED_MODEL
from vendors.pinecone_client import INDEX
from datetime import datetime, timezone
import os, math

# NEW: use helpers from memory/selection.py
from memory.selection import hybrid_rank, pack_to_budget, one_hop_graph

TOPK_PER_TYPE = int(os.getenv("TOPK_PER_TYPE", "30"))
RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))

def embed(text: str):
    text = text[:8000]
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def upsert_memory_vector(mem_id: str, user_id: str | None, type_: str, content: str, title: str, tags, importance: int,
                          created_at_iso: str | None = None, source: str | None = None, role_view=None, entity_ids=None):
    vec = embed(content)
    meta = {
        "memory_id": mem_id,
        "user_id": str(user_id) if user_id else None,
        "type": type_,
        "title": (title or "")[:120],
        "tags": tags or [],
        "importance": int(importance),
        "created_at": created_at_iso or _now_iso(),
        "source": source or None,
        "role_view": role_view or [],
        "entity_ids": entity_ids or [],
        "deleted": False,
    }
    vid = f"mem_{mem_id}" if not str(mem_id).startswith("mem_") else str(mem_id)
    INDEX.upsert(vectors=[{"id": vid, "values": vec, "metadata": meta}], namespace=type_)

def search_per_type(user_id: str | None, query: str, types=None, top_k:int = TOPK_PER_TYPE):
    vec = embed(query)
    types = types or ["episodic","semantic","procedural"]
    results = []
    flt = {"deleted": False}
if user_id: flt["user_id"] = str(user_id)
if role:
    flt["$or"] = [
        {"role_view": {"$eq": []}},
        {"role_view": {"$contains": role}}
    ]

    for t in types:
        res = INDEX.query(vector=vec, top_k=top_k, include_metadata=True, filter=flt, namespace=t)
        for m in (res.matches or []):
            md = m.metadata or {}
            md["__namespace"] = t
            md["__score_semantic"] = float(m.score or 0.0)
            results.append(md)
    return results

# Backward-compatible search
def search(user_id: str | None, query: str, top_k=6, types=None):
    res = search_per_type(user_id, query, types=types or ["semantic"], top_k=top_k)
    class M: pass
    out = []
    for md in res:
        m = M()
        m.score = md.get("__score_semantic", 0.0)
        m.metadata = md
        out.append(m)
    return out
