"""
agent/retrieval.py
------------------
Role-aware, tag-aware retrieval across type namespaces (episodic|semantic|procedural).
- Pinecone v5+ compatible (with legacy fallback)
- OpenAI embeddings via vendors.openai_client
- Merges per-type results, fetches full memory rows (title, text, tags) from store
- Provides upsert_memory_vector for new/updated memories
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ------------------------
# Optional dependencies
# ------------------------
try:
    from vendors.openai_client import client as _oai_client
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
except Exception:
    _oai_client = None
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Pinecone v5 prefers Pinecone(...) and .Index / .Indexes; support legacy too.
_pc = None
_pc_index = None

def _pinecone_index():
    global _pc, _pc_index
    if _pc_index is not None:
        return _pc_index
    index_name = os.getenv("PINECONE_INDEX", "uap-kb")
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing")
    try:
        # v5
        from pinecone import Pinecone
        _pc = Pinecone(api_key=api_key)
        _pc_index = _pc.Index(index_name)
        return _pc_index
    except Exception as ex:
        logger.warning("Pinecone v5 init failed, trying legacy: %s", ex)
        try:
            import pinecone
            pinecone.init(api_key=api_key, environment=os.getenv("PINECONE_ENV"))
            _pc_index = pinecone.Index(index_name)
            return _pc_index
        except Exception as ex2:
            raise RuntimeError(f"Failed to init Pinecone: {ex2}")

# ------------------------
# Store adapter (optional)
# ------------------------
try:
    from agent import store
except Exception:
    store = None

def _embed(text: str) -> List[float]:
    if _oai_client is None:
        raise RuntimeError("OpenAI client unavailable for embeddings")
    resp = _oai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding  # type: ignore

def _build_filter(role: Optional[str], tags_any: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    flt: Dict[str, Any] = {}
    ors = []

    if role:
        ors.append({"role_view": {"$in": [role]}})
    # NOTE: Pinecone filter cannot reliably test empty arrays/no field across all versions.
    # We keep it simple—if role not provided, we don't filter by it.

    if tags_any:
        flt["tags"] = {"$in": tags_any}

    if not flt and not ors:
        return None
    if ors and flt:
        return {"$and": [flt, {"$or": ors}]}
    if ors:
        return {"$or": ors}
    return flt

def _fetch_memories(mem_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dict id -> record {id,title,text,tags,type,created_at,source,entity_ids}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not mem_ids:
        return out
    if store is None:
        # best-effort: return skeletons
        for mid in mem_ids:
            out[mid] = {"id": mid, "title": None, "text": "", "tags": [], "type": None, "source": None}
        return out
    try:
        if hasattr(store, "memories_by_ids"):
            rows = store.memories_by_ids(mem_ids)
        elif hasattr(store, "fetch_memories_by_ids"):
            rows = store.fetch_memories_by_ids(mem_ids)
        else:
            rows = []
    except Exception as ex:
        logger.warning("fetch_memories_by_ids failed: %s", ex)
        rows = []
    for r in rows or []:
        out[r["id"]] = {
            "id": r["id"],
            "title": r.get("title"),
            "text": r.get("text") or r.get("summary") or "",
            "tags": r.get("tags") or [],
            "type": r.get("type"),
            "created_at": r.get("created_at"),
            "source": r.get("source"),
            "entity_ids": r.get("entity_ids") or [],
        }
    return out

def retrieve(*, query: str, role: Optional[str], session_id: Optional[str], top_k: int = 30,
             types: Optional[List[str]] = None, tags_any: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Query Pinecone per-type namespace, apply metadata filters, merge results,
    and return full memory records + score.
    """
    idx = _pinecone_index()
    vec = _embed(query)
    namespaces = types or ["episodic","semantic","procedural"]
    flt = _build_filter(role, tags_any)

    merged: List[Tuple[float, Dict[str, Any]]] = []

    for ns in namespaces:
        try:
            # v5 query signature
            res = idx.query(
                vector=vec, top_k=top_k, namespace=ns, include_metadata=True, filter=flt
            )
            matches = getattr(res, "matches", []) or res.get("matches", [])  # support dict-like
            for m in matches:
                mid = m.get("id") or m.id  # type: ignore
                score = float(m.get("score") if isinstance(m, dict) else m.score)  # type: ignore
                md = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})  # type: ignore
                merged.append((score, {
                    "id": mid,
                    "score": score,
                    "type": ns,
                    "title": md.get("title"),
                    "tags": md.get("tags") or [],
                    "source": md.get("source"),
                    "created_at": md.get("created_at"),
                    "entity_ids": md.get("entity_ids") or [],
                }))
        except TypeError:
            # legacy client
            res = idx.query(vector=vec, top_k=top_k, namespace=ns, include_metadata=True, filter=flt)
            for m in res.matches:  # type: ignore
                merged.append((float(m.score), {
                    "id": m.id, "score": float(m.score), "type": ns,
                    "title": (m.metadata or {}).get("title"),
                    "tags": (m.metadata or {}).get("tags") or [],
                    "source": (m.metadata or {}).get("source"),
                    "created_at": (m.metadata or {}).get("created_at"),
                    "entity_ids": (m.metadata or {}).get("entity_ids") or [],
                }))
        except Exception as ex:
            logger.warning("pinecone query failed for ns=%s: %s", ns, ex)

    # sort desc by score, stable
    merged.sort(key=lambda t: t[0], reverse=True)
    toplist = [r for _, r in merged]

    # fetch full memory rows to add text/title if missing
    id_list = [r["id"] for r in toplist]
    id2mem = _fetch_memories(id_list)
    out: List[Dict[str, Any]] = []
    for r in toplist:
        mid = r["id"]
        full = id2mem.get(mid, {})
        # fill gaps
        r["title"] = r.get("title") or full.get("title")
        r["text"] = full.get("text", "")
        r["tags"] = r.get("tags") or full.get("tags") or []
        r["source"] = r.get("source") or full.get("source")
        out.append(r)
    return out

def upsert_memory_vector(*, mem_id: str, user_id: Optional[str], type: str, content: str,
                         title: Optional[str], tags: Optional[List[str]], importance: int,
                         created_at_iso: Optional[str], source: Optional[str],
                         role_view: Optional[List[str]], entity_ids: Optional[List[str]]) -> None:
    """
    Upsert one memory vector into Pinecone with PRD metadata.
    Vector id: mem_{mem_id}
    Namespace: type (episodic|semantic|procedural)
    """
    idx = _pinecone_index()
    vec = _embed(content or (title or ""))
    vector_id = f"mem_{mem_id}" if not str(mem_id).startswith("mem_") else str(mem_id)
    md = {
        "type": type,
        "title": title,
        "tags": tags or [],
        "created_at": created_at_iso,
        "role_view": role_view or [],
        "entity_ids": entity_ids or [],
        "source": source or "chat",
    }
    try:
        # v5
        idx.upsert(
            vectors=[{"id": vector_id, "values": vec, "metadata": md}],
            namespace=type
        )
    except TypeError:
        # legacy
        idx.upsert(vectors=[(vector_id, vec, md)], namespace=type)
    except Exception as ex:
        logger.error("pinecone upsert failed: %s", ex)
        raise
