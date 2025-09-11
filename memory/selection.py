from __future__ import annotations

import time, math, os
from typing import Any, Dict, List, Optional

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS","2000"))

def _recency_score(created_at: Optional[str], half_life_days: int = int(os.getenv("RECENCY_HALFLIFE_DAYS","90"))) -> float:
    if not created_at:
        return 0.5
    try:
        from datetime import datetime, timezone
        t = datetime.fromisoformat(created_at.replace("Z","+00:00"))
        days = (datetime.now(timezone.utc)-t).total_seconds()/86400.0
        return math.exp(-math.log(2)*(days/half_life_days))
    except Exception:
        return 0.5

def cross_layer_boost(chunks):
    boosted = []
    for i, c1 in enumerate(chunks):
        for j, c2 in enumerate(chunks):
            if i >= j:
                continue
            # If they come from different layers but share entity_ids
            if c1["type"] != c2["type"] and set(c1["entity_ids"]) & set(c2["entity_ids"]):
                c1["score"] += 0.1
                c2["score"] += 0.1
        boosted.append(c1)
    return boosted


def rank_and_pack_minimal(hits: List[Dict[str,Any]], records: List[Dict[str,Any]], wm_msgs: List[Dict[str,Any]], prompt: str) -> Dict[str,Any]:
    # Map embedding_id -> record
    rec_by_emb = { (r.get("embedding_id") or r.get("id")): r for r in records }
    scored=[]
    for h in hits:
        rec = rec_by_emb.get(h.get("id"))
        if not rec: 
            continue
        sem = h.get("score", 0.0)
        recency = _recency_score(rec.get("created_at"))
        score = 0.7*sem + 0.3*recency
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    # Build context (rough token estimate by chars)
    ctx=[]
    budget_chars = MAX_CONTEXT_TOKENS*4
    used=0
    for _, rec in scored:
        txt = (rec.get("text") or "")[:4000]
        if used + len(txt) > budget_chars:
            break
        ctx.append({"id": rec.get("id"), "title": rec.get("title"), "text": txt})
        used += len(txt)
    return {"context": ctx, "ranked_ids": [r.get("id") for _, r in scored]}
