"""
memory/selection.py
-------------------
PRD §7 selection utilities:
- hybrid_rank(): 0.55*semantic + 0.20*recency + 0.15*entity_overlap + 0.10*source_priority
- pack_to_budget(): keep last 2 turns + top-ranked chunks to MAX_CONTEXT_TOKENS; drop near-dups
- one_hop_graph(): optional expansion via store if available; safe to no-op
"""

from __future__ import annotations

import os
import re
import math
import time
from typing import Any, Dict, List, Optional, Tuple

RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))

try:
    from agent import store
except Exception:
    store = None

def _approx_tokens(s: str) -> int:
    return max(1, int(len(s or "")/4))

def _parse_iso(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        # very lenient: treat as UNIX seconds if digits only
        if ts.isdigit():
            return float(ts)
    except Exception:
        pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts.replace("Z","")).timestamp()
    except Exception:
        return None

def _recency_weight(created_at: Optional[str]) -> float:
    ts = _parse_iso(created_at)
    if not ts:
        return 0.0
    days = max(0.0, (time.time() - ts) / 86400.0)
    hl = float(RECENCY_HALFLIFE_DAYS)
    return math.exp(-math.log(2) * days / hl)

def _query_entities(q: str) -> List[str]:
    # naive entity extraction: words >= 4 chars, alpha-numeric, de-duplicated
    toks = re.findall(r'[A-Za-z0-9][A-Za-z0-9\-]{3,}', q.lower())
    uniq = []
    seen = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:12]

_SOURCE_PRI = {
    "minutes": 1.0,
    "sop": 0.9,
    "wiki": 0.7,
    "transcript": 0.6,
    "misc": 0.5,
    None: 0.5,
}

def _source_priority(r: Dict[str, Any]) -> float:
    src = r.get("source")
    # allow tags to hint source type
    tags = [t.lower() for t in (r.get("tags") or [])]
    if "minutes" in tags:
        return 1.0
    if "sop" in tags:
        return 0.9
    if "wiki" in tags:
        return 0.7
    if "transcript" in tags:
        return 0.6
    return _SOURCE_PRI.get(src, 0.5)

def _entity_overlap(q_ents: List[str], r_ents: List[str]) -> float:
    if not q_ents or not r_ents:
        return 0.0
    A = set(q_ents)
    B = set([e.lower() for e in r_ents])
    inter = len(A & B)
    union = len(A | B)
    return float(inter) / float(union) if union else 0.0

def hybrid_rank(candidates: List[Dict[str, Any]], query: str, weights: Optional[Dict[str,float]] = None) -> List[Dict[str, Any]]:
    w = {"semantic":0.55, "recency":0.20, "entity":0.15, "source":0.10}
    if weights:
        w.update(weights)
    qents = _query_entities(query)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in candidates:
        sem = float(r.get("score", 0.0))
        rec = _recency_weight(r.get("created_at"))
        ent = _entity_overlap(qents, r.get("entity_ids") or [])
        src = _source_priority(r)
        s = w["semantic"]*sem + w["recency"]*rec + w["entity"]*ent + w["source"]*src
        scored.append((s, r))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored]

def pack_to_budget(history: List[Dict[str, Any]], ranked: List[Dict[str, Any]], budget_tokens: int) -> Dict[str, Any]:
    parts: List[str] = []
    # keep last 2 user+assistant turns if history provided as dicts with {role, content}
    if history:
        tail = history[-4:]
        for turn in tail:
            role = turn.get("role","user")
            content = turn.get("content") or turn.get("text") or ""
            parts.append(f"[{role}] {content}".strip())
    used = _approx_tokens("\n".join(parts))

    seen_hashes = set()
    for r in ranked:
        txt = r.get("text") or ""
        if not txt:
            continue
        norm = " ".join(txt.lower().split())
        h = hash(norm)
        if h in seen_hashes:
            continue
        t = _approx_tokens(txt)
        if used + t > budget_tokens:
            continue
        title = r.get("title") or ""
        parts.append(f"[{r.get('type','?')}:{r.get('id')}] {title}\n{txt}")
        used += t
        seen_hashes.add(h)

    return {"context": "\n\n".join(parts)}

def one_hop_graph(records: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Optional expansion via store: find additional memories connected by shared entity_ids.
    """
    try:
        if store and hasattr(store, "memories_by_entity_overlap"):
            # expects a list of entity_ids and returns more memories (dicts with id,text,...)
            ent_ids = set()
            for r in records:
                for e in r.get("entity_ids") or []:
                    ent_ids.add(e)
            if not ent_ids:
                return []
            return store.memories_by_entity_overlap(list(ent_ids), limit=limit) or []
    except Exception:
        pass
    return []
