# memory/autosave.py
# Implements autosave-from-chat policy and persistence

import os, time, hashlib
from typing import List, Dict, Any, Optional
from agent import store
from agent.retrieval import upsert_memory_vector

AUTOSAVE_CONF_THRESHOLD = float(os.getenv("AUTOSAVE_CONF_THRESHOLD", "0.75"))
AUTOSAVE_ENTITY_CONF_THRESHOLD = float(os.getenv("AUTOSAVE_ENTITY_CONF_THRESHOLD", "0.85"))

def normalize(text: str) -> str:
    return " ".join(text.split()).strip()

def sha256_normalized(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()

def autosave_from_candidates(candidates: List[Dict[str, Any]], session_id: Optional[str]) -> Dict[str, Any]:
    saved_items = []
    t0 = time.time()
    for c in candidates or []:
        fact_type = c.get("fact_type")
        conf = float(c.get("confidence", 0.0))
        if fact_type in ("decision","deadline","procedure") and conf >= AUTOSAVE_CONF_THRESHOLD:
            text = normalize(c.get("text",""))
            if not text:
                continue
            dh = sha256_normalized(text)
            if store.find_memory_by_dedupe_hash(dh):
                continue
            title = c.get("title") or text[:120]
            mem_type = "episodic" if fact_type in ("decision","deadline") else "procedural"
            tags = c.get("tags") or []
            mem_id = store.insert_memory(type=mem_type, title=title, text=text, tags=tags,
                                         source="chat", file_id=None, session_id=session_id,
                                         author_user_id=None, role_view=None, dedupe_hash=dh)
            try:
                upsert_memory_vector(mem_id, user_id=None, type_=mem_type, content=text, title=title,
                                     tags=tags, importance=1, created_at_iso=None, source="chat",
                                     role_view=None, entity_ids=[])
                emb_id = f"mem_{mem_id}"
                store.update_memory_embedding_id(mem_id, emb_id)
            except Exception as ex:
                store.log_tool_run("autosave_embed", {"memory_id": mem_id}, {"error": str(ex)}, False, int((time.time()-t0)*1000))
            saved_items.append({"memory_id": mem_id, "type": mem_type, "title": title})
        elif fact_type == "entity" and conf >= AUTOSAVE_ENTITY_CONF_THRESHOLD:
            text = normalize(c.get("text",""))
            if not text:
                continue
            # entity autosave simplified: just log, not persisted
            store.log_tool_run("autosave_entity_skip", {"entity":c}, {"reason":"not implemented for autosave"}, True, 0)
    return {"saved": bool(saved_items), "items": saved_items}
