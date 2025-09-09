# router/search.py
from typing import Optional, List, Dict, Any
import os, re, math, datetime
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# ----- request/response schema -----

class SearchIn(BaseModel):
    q: str = Field(..., description="User query text")
    type: Optional[List[str]] = Field(default=None, description="Filter types: episodic, semantic, procedural")
    top_k: int = Field(default=12, ge=1, le=50)
    include_text: bool = Field(default=True)

def _norm_q(s: str) -> str:
    s = s or ""
    s = s.strip()
    # light normalization to improve recall
    s = re.sub(r"\s+", " ", s)
    return s

def _resp_data(resp) -> List[Dict[str, Any]]:
    if resp is None: return []
    if hasattr(resp, "data"):  # supabase-py v2
        return resp.data or []
    if isinstance(resp, dict):
        return resp.get("data") or []
    return []

# --- hybrid rank helpers (Phase 3) ---
HALF_LIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))

def _recency_weight(iso: str) -> float:
    """Exponential decay with half-life in days (default 90)."""
    try:
        dt = datetime.datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
        delta = datetime.datetime.now(datetime.timezone.utc) - dt
        days = max(0.0, delta.total_seconds() / 86400.0)
        return math.exp(-math.log(2) * (days / max(1, HALF_LIFE_DAYS)))
    except Exception:
        return 1.0

def _source_priority(title: Optional[str], typ: Optional[str]) -> float:
    """
    Simple source prior:
      meeting minutes > SOP/policy > wiki > misc
      plus a small bump if type is procedural (SOP-like).
    """
    t = (title or "").lower()
    if "minutes" in t or "meeting" in t:
        return 1.0
    if "sop" in t or "policy" in t or (typ or "").lower() == "procedural":
        return 0.9
    if "wiki" in t:
        return 0.7
    return 0.5


# ----- main endpoint -----

@router.post("/search/semantic")
def search_semantic(body: SearchIn, x_api_key: Optional[str] = Header(None)):
    # Optional API key
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    q = _norm_q(body.q)
    if not q:
        raise HTTPException(status_code=400, detail="Query 'q' must be non-empty")

    # Decide namespaces to search
    allowed = {"episodic","semantic","procedural"}
    types = [t for t in (body.type or ["episodic","semantic","procedural"]) if t in allowed]
    if not types:
        types = ["episodic","semantic","procedural"]

    # 1) Create embedding for the query
    try:
        from openai import OpenAI
        oai = OpenAI()
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        kwargs = {"model": embed_model, "input": q}
        if os.getenv("EMBED_DIM"):
            kwargs["dimensions"] = int(os.getenv("EMBED_DIM"))
        eresp = oai.embeddings.create(**kwargs)
        qvec = eresp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # 2) Query Pinecone per namespace
    try:
        from vendors.pinecone_client import get_index
        index = get_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone index error: {e}")

    matches: List[Dict[str, Any]] = []
    for t in types:
        try:
            # SDK compatibility: both dict and object responses supported
            res = index.query(
                vector=qvec,
                top_k=body.top_k,
                namespace=t,
                include_values=False,
                include_metadata=True,
            )
            items = []
            if isinstance(res, dict):
                items = res.get("matches", []) or []
            else:
                items = getattr(res, "matches", []) or []

            for m in items:
                mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
                score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
                md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
                if not mid: 
                    continue
                # our vectors are named mem_<uuid>
                memory_id = mid[4:] if mid.startswith("mem_") else mid
                title = (md or {}).get("title")
                matches.append({
                    "memory_id": memory_id,
                    "title": title,
                    "type": t,
                    "score": score
                })
        except Exception:
            # keep searching other namespaces
            continue

    if not matches:
        return {"items": [], "note": "no_matches"}

    # dedupe by memory_id, keep best score
    best: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        k = m["memory_id"]
        if k not in best or (m.get("score") or 0) > (best[k].get("score") or 0):
            best[k] = m
    ordered = sorted(best.values(), key=lambda x: (x.get("score") or 0), reverse=True)
    ordered = ordered[:body.top_k]

    # 3) Join full rows from Supabase
    try:
        from vendors.supabase_client import get_client
        sb = get_client()
        ids = [o["memory_id"] for o in ordered]
        cols = "id,type,title,tags,created_at,embedding_id"
        text_col = (os.getenv("MEMORIES_TEXT_COLUMN") or "text").strip().lower()
        if body.include_text:
            cols += f",{text_col}"
        # Build OR filter
        qsb = sb.table("memories").select(cols)
        # supabase-py doesn't have .in_ for all versions; use or_ with eq chains
        if ids:
            ors = ",".join([f"id.eq.{i}" for i in ids])
            qsb = qsb.or_(ors)
        rs = qsb.execute()
        rows = _resp_data(rs)
        # index by id
        by_id = {r["id"]: r for r in rows} if rows else {}
        result = []
        for o in ordered:
            r = by_id.get(o["memory_id"])
            if not r:
                result.append({**o, "found": False})
                continue
            item = {
                "id": r["id"],
                "type": r["type"],
                "title": r.get("title"),
                "tags": r.get("tags") or [],
                "created_at": r.get("created_at"),
                "score": o.get("score"),
            }
            if body.include_text:
                item["text"] = r.get(text_col)
            result.append(item)
                # --- Phase 3: Hybrid re-ranking ---
        # score_final = 0.55*semantic + 0.20*recency + 0.10*source_priority
        for it in result:
            base = it.get("score") or 0.0
            rec = _recency_weight(it.get("created_at") or "")
            sp  = _source_priority(it.get("title"), it.get("type"))
            it["hybrid_score"] = 0.55*base + 0.20*rec + 0.10*sp  # reserve 0.15 for entity overlap (Phase 4)
        result.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        return {"items": result}
    except Exception as e:
        # If join fails, still return match skeletons
        return {"items": ordered, "warning": f"join_failed: {e}"}
