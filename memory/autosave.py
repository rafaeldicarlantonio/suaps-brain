from __future__ import annotations

import os, time
from typing import Any, Dict, List
from vendors.openai_client import EMBED_MODEL, client
from vendors.pinecone_client import get_index
from vendors.supabase_client import supabase
from agent import store

AUTOSAVE_CONF_THRESHOLD = float(os.getenv("AUTOSAVE_CONF_THRESHOLD","0.75"))
AUTOSAVE_ENTITY_CONF_THRESHOLD = float(os.getenv("AUTOSAVE_ENTITY_CONF_THRESHOLD","0.85"))

def _embed(text: str):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def autosave_from_candidates(items: List[Dict[str,Any]], session_id: str) -> Dict[str,Any]:
    saved_items=[]
    for it in items or []:
        fact_type = it.get("fact_type")
        conf = float(it.get("confidence",0.0))
        if fact_type in ("decision","deadline","procedure") and conf >= AUTOSAVE_CONF_THRESHOLD:
            text = it.get("text","").strip()
            if not text:
                continue
            import hashlib
            dh = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ex = store.find_memory_by_dedupe_hash(dh)
            if ex:
                continue
            mem = {
                "type": "episodic" if fact_type in ("decision","deadline") else "procedural",
                "title": it.get("title") or fact_type,
                "text": text,
                "tags": it.get("tags") or [],
                "source": "chat",
                "session_id": session_id,
                "dedupe_hash": dh,
            }
            row = store.upsert_memory(mem)
            # Embed + upsert
            vec = _embed(text)
            idx = get_index()
            emb_id = f"mem_{row['id']}"
            idx.upsert(vectors=[{"id": emb_id, "values": vec, "metadata": {"type": mem["type"], "title": mem["title"], "tags": mem["tags"], "source": "chat"}}], namespace=mem["type"])
            store.update_memory_embedding_id(row["id"], emb_id)
            saved_items.append({"memory_id": row["id"], "type": mem["type"], "title": mem["title"]})
    return {"saved": len(saved_items)>0, "items": saved_items}
