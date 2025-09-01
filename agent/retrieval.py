from vendors.openai_client import client, EMBED_MODEL
from vendors.pinecone_client import INDEX
from datetime import datetime, timezone
import os, math

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

def _recency_score(created_at_iso: str | None) -> float:
    try:
        if not created_at_iso: return 0.5
        dt = datetime.fromisoformat(created_at_iso.replace("Z","+00:00"))
        days = (datetime.now(timezone.utc) - dt).days
        return math.exp(-math.log(2) * (days / RECENCY_HALFLIFE_DAYS))
    except Exception:
        return 0.5

def _entity_overlap(q_entities: set, e_ids: list) -> float:
    if not q_entities or not e_ids: return 0.0
    s = set(e_ids)
    inter = len(q_entities & s)
    union = len(q_entities | s)
    return (inter / union) if union else 0.0

def _source_priority(src: str | None) -> float:
    m = {"minutes":1.0, "sop":0.9, "wiki":0.7, "transcript":0.6}
    return m.get((src or "").lower(), 0.5)

def search_per_type(user_id: str | None, query: str, types=None, top_k:int = TOPK_PER_TYPE):
    vec = embed(query)
    types = types or ["episodic","semantic","procedural"]
    results = []
    flt = {"deleted": False}
    if user_id: flt["user_id"] = str(user_id)
    for t in types:
        res = INDEX.query(vector=vec, top_k=top_k, include_metadata=True, filter=flt, namespace=t)
        for m in (res.matches or []):
            md = m.metadata or {}
            md["__namespace"] = t
            md["__score_semantic"] = float(m.score or 0.0)
            results.append(md)
    return results

def hybrid_rank(query: str, candidates: list[dict], q_entities: set | None = None):
    q_entities = q_entities or set()
    scored = []
    for md in candidates:
        sem = float(md.get("__score_semantic", 0.0))
        rec = _recency_score(md.get("created_at"))
        ent = _entity_overlap(q_entities, md.get("entity_ids") or [])
        src = _source_priority(md.get("source"))
        total = 0.55*sem + 0.20*rec + 0.15*ent + 0.10*src
        md2 = dict(md)
        md2["__score_hybrid"] = total
        scored.append(md2)
    scored.sort(key=lambda x: x["__score_hybrid"], reverse=True)
    return scored

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
