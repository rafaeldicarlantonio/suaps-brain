import os
import datetime
from typing import List, Dict, Any, Optional

from ingest.pipeline import normalize_text, upsert_memories_from_chunks
from memory.autosave_classifier import classify_importance


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()


def _save_memory(
    sb,
    pinecone_index,
    candidate: Dict[str, Any],
    session_id: Optional[str],
    text_col_env: str,
    author_user_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Persist a single fact/procedure memory via the ingest pipeline.
    Returns the inserted/updated memory dict or None if no change.
    """
    ftype = candidate.get("fact_type") or "note"
    title = candidate.get("title") or ftype.title()
    text = normalize_text(candidate.get("text") or "")
    tags = candidate.get("tags") or []

    r = upsert_memories_from_chunks(
        sb=sb,
        pinecone_index=pinecone_index,
        embedder=None,
        file_id=None,
        title_prefix=title,
        chunks=[text],
        mem_type=ftype if ftype in ("episodic", "procedural") else "episodic",
        tags=tags,
        role_view=[],
        author_user_id=author_user_id,
        metadata_overrides={"saved_at": _now_iso()},
    )
    if r.get("upserted") or r.get("updated"):
        return r.get("upserted") or r.get("updated")[0]
    return None


# --- REPLACE your apply_autosave() with this version ---

def apply_autosave(
    *,
    sb,
    pinecone_index,
    candidates: List[Dict[str, Any]],
    session_id: Optional[str],
    text_col_env: str = "value",
    author_user_id=None,
) -> Dict[str, Any]:
    """
    Process autosave candidates:
      • LLM-based importance classification.
      • Entity-specific thresholding (stricter than general facts).
      • Splits 'review' vs 'skipped' so borderline items aren't lost.
      • Saves high-quality memories via the ingest pipeline.
    """
    thr_fact = float(os.getenv("AUTOSAVE_CONF_THRESHOLD", "0.75"))
    thr_ent  = float(os.getenv("AUTOSAVE_ENTITY_CONF_THRESHOLD", "0.85"))

    saved: List[Dict[str, Any]] = []
    review: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for c in (candidates or []):
        ftype = (c.get("fact_type") or "").lower()           # decision|deadline|procedure|entity
        conf  = float(c.get("confidence") or 0.0)
        text  = normalize_text(c.get("text") or "")
        title = c.get("title") or ftype.title()
        tags  = c.get("tags") or []

        if not text or not ftype:
            skipped.append({"reason": "missing_fields", "title": title})
            continue

        # 🔎 Importance classification (adds c.importance & c.importance_score)
        imp = classify_importance(c)
        c["importance"] = imp.get("importance", "low")
        c["importance_score"] = float(imp.get("importance_score", 0.0))

        # --- Threshold policy ---
        # entities need higher confidence than general facts
        # general facts gate at thr_fact
        if ftype == "entity":
            if c["importance"] == "high" and conf >= thr_ent:
                mem = _save_memory(sb, pinecone_index, c, session_id, text_col_env, author_user_id)
                if mem: saved.append(mem)
            elif c["importance"] == "medium" or (0.6 <= conf < thr_ent):
                c["review_required"] = True
                review.append(c)
            else:
                skipped.append({"reason": "low_conf_entity", "title": title})
        else:
            if c["importance"] == "high" and conf >= thr_fact:
                mem = _save_memory(sb, pinecone_index, c, session_id, text_col_env, author_user_id)
                if mem: saved.append(mem)
            elif c["importance"] == "medium" or (0.6 <= conf < thr_fact):
                c["review_required"] = True
                review.append(c)
            else:
                skipped.append({"reason": f"low_conf_{ftype}", "title": title})

    return {
        "saved": bool(saved),
        "items": saved,
        "review": review,     # ⬅️ NEW: borderline items separated from 'skipped'
        "skipped": skipped,
    }

