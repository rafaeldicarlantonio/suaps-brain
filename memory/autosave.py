# memory/autosave.py
import os, hashlib, datetime
from typing import List, Dict, Any, Optional

from ingest.pipeline import normalize_text, upsert_memories_from_chunks

def _now_iso():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def apply_autosave(
    *,
    sb,
    pinecone_index,
    candidates: List[Dict[str, Any]],
    session_id: Optional[str],
    text_col_env: str = "value",
) -> Dict[str, Any]:
    """Filter autosave candidates by thresholds, save them via the ingest pipeline."""
    thr_fact = float(os.getenv("AUTOSAVE_CONF_THRESHOLD", "0.75"))
    thr_ent  = float(os.getenv("AUTOSAVE_ENTITY_CONF_THRESHOLD", "0.85"))

    saved, skipped = [], []
    for c in candidates or []:
        ftype = (c.get("fact_type") or "").lower()
        conf  = float(c.get("confidence") or 0)
        title = c.get("title") or ftype.title()
        text  = normalize_text(c.get("text") or "")
        tags  = c.get("tags") or []

        if not text or not ftype:
            skipped.append({"reason": "missing_fields", "title": title})
            continue

        # policy: entities require higher confidence; other facts use thr_fact
        if ftype == "entity" and conf < thr_ent:
            skipped.append({"reason": "low_conf_entity", "title": title})
            continue
        if ftype != "entity" and conf < thr_fact:
            skipped.append({"reason": "low_conf_fact", "title": title})
            continue

        # map type to memory.type
        mem_type = "procedural" if ftype == "procedure" else "episodic"

        # upsert via pipeline (single "chunk")
        r = upsert_memories_from_chunks(
            sb=sb,
            pinecone_index=pinecone_index,
            embedder=None,
            file_id=None,
            title_prefix=title,
            chunks=[text],
            mem_type=mem_type,
            tags=tags,
            role_view=[],
            source="chat",
            text_col_env=text_col_env,
        )
        if r.get("upserted") or r.get("updated"):
            item = (r.get("upserted") or r.get("updated"))[0]
            saved.append({"memory_id": item.get("memory_id"), "type": mem_type, "title": title})
        else:
            skipped.append({"reason": "pipeline_noop", "title": title})

    return {"saved": True if saved else False, "items": saved, "skipped": skipped}
