# memory/selection.py
# Implements hybrid ranking, one-hop graph expansion, and context packing

import math, os
from datetime import datetime, timezone
from typing import List, Dict, Any

RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))

def _recency_score(created_at_iso: str) -> float:
    try:
        if not created_at_iso:
            return 0.5
        dt = datetime.fromisoformat(created_at_iso.replace("Z","+00:00"))
        days = (datetime.now(timezone.utc)-dt).days
        return math.exp(-math.log(2) * (days/RECENCY_HALFLIFE_DAYS))
    except Exception:
        return 0.5

def _entity_overlap(q_entities:set, e_ids:List[str]) -> float:
    if not q_entities or not e_ids:
        return 0.0
    s = set(e_ids)
    return len(q_entities & s)/len(q_entities | s) if (q_entities|s) else 0.0

def _source_priority(src: str) -> float:
    mapping = {"minutes":1.0,"sop":0.9,"wiki":0.7,"transcript":0.6}
    return mapping.get((src or "").lower(),0.5)

def hybrid_rank(query: str, candidates: List[Dict[str,Any]], q_entities:set|None=None) -> List[Dict[str,Any]]:
    q_entities = q_entities or set()
    scored = []
    for md in candidates:
        sem = float(md.get("__score_semantic",0.0))
        rec = _recency_score(md.get("created_at"))
        ent = _entity_overlap(q_entities, md.get("entity_ids") or [])
        src = _source_priority(md.get("source"))
        total = 0.55*sem + 0.20*rec + 0.15*ent + 0.10*src
        md2=dict(md)
        md2["__score_hybrid"]=total
        scored.append(md2)
    scored.sort(key=lambda x:x["__score_hybrid"], reverse=True)
    return scored

def one_hop_graph(records: List[Dict[str,Any]], db=None, limit:int=10) -> List[Dict[str,Any]]:
    # db is supabase client wrapper (agent.store.supabase)
    if not db or not records:
        return []
    ids = []
    for r in records:
        for e in r.get("entity_ids") or []:
            ids.append(e)
    if not ids:
        return []
    # fetch other memories linked to these entities
    out = []
    for e in ids[:limit]:
        try:
            rows = db.table("entity_mentions").select("memory_id").eq("entity_id", e).limit(limit).execute().data
            for row in rows or []:
                mids = row.get("memory_id")
                if mids:
                    mems = db.table("memories").select("*").eq("id", mids).limit(1).execute().data
                    for m in mems or []:
                        out.append(m)
        except Exception:
            continue
    return out

def pack_to_budget(history, ranked, budget=MAX_CONTEXT_TOKENS):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    ctx, used = list(history), sum(len(enc.encode(h.get("content",""))) for h in history if h.get("content"))
    kept = []
    for r in ranked:
        t = (r.get("text") or r.get("content") or "")
        need = len(enc.encode(t))
        if used + need > budget: break
        # cosine dedupe
        if not is_near_duplicate(r, kept):  # implement cosine>0.98 check using stored embeddings if available
            kept.append(r)
            ctx.append(r)
            used += need
    return ctx

