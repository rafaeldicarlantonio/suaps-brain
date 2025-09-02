"""
agent/retrieval.py
------------------
Retrieval helpers with PRD-compliant role filters and per-type querying.
Import-safe and tolerant of missing dependencies.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional  # <-- ensure Optional is defined

logger = logging.getLogger(__name__)

# Tolerant imports
try:
    from vendors.openai_client import client as _oai_client
except Exception:
    _oai_client = None

try:
    from vendors.openai_client import EMBED_MODEL as _EMBED_MODEL
except Exception:
    _EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

try:
    # Pinecone 5.x client expected; expose Index via pc.Index(name)
    from vendors.pinecone_client import pc as _pc
except Exception:
    _pc = None

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "uap-kb")
TOPK_PER_TYPE = int(os.getenv("TOPK_PER_TYPE", "30"))

# ----------------- embedding -----------------

def embed_text(text: str) -> List[float]:
    if not text:
        return []
    if _oai_client is None:
        logger.warning("OpenAI client unavailable; embed_text returns empty vector")
        return []
    try:
        resp = _oai_client.embeddings.create(model=_EMBED_MODEL, input=text[:8000])
        return resp.data[0].embedding
    except Exception as ex:
        logger.warning("embed_text failed: %s", ex)
        return []

# ----------------- filters -----------------

def _role_filter(role: Optional[str]) -> Dict[str, Any]:
    if not role:
        return {}
    # allow docs that are public (role_view empty) OR contain the role
    return {"$or": [{"role_view": {"$eq": []}}, {"role_view": {"$contains": role}}]}

def build_metadata_filter(*, role: Optional[str], tags_any: Optional[List[str]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    conds: List[Dict[str, Any]] = [{"deleted": False}]
    if tags_any:
        tags = [t for t in tags_any if isinstance(t, str)]
        if tags:
            conds.append({"tags": {"$in": tags}})
    rf = _role_filter(role)
    if rf:
        conds.append(rf)
    if extra:
        conds.append(extra)
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}

# ----------------- pinecone helpers -----------------

def _index():
    if _pc is None:
        logger.warning("pinecone client unavailable")
        return None
    try:
        return _pc.Index(PINECONE_INDEX)
    except Exception as ex:
        logger.warning("pinecone index init failed: %s", ex)
        return None

def _query_namespace(index, vector, namespace: str, top_k: int, flt: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        res = index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=flt or None,
            include_metadata=True,
        )
        out: List[Dict[str, Any]] = []
        matches = getattr(res, "matches", []) or []
        for m in matches:
            md = getattr(m, "metadata", {}) or {}
            out.append({
                "id": md.get("memory_id") or getattr(m, "id", None),
                "score": float(getattr(m, "score", 0.0) or 0.0),
                "type": md.get("type") or namespace,
                "title": md.get("title"),
                "text": md.get("text") or md.get("summary"),
                "tags": md.get("tags") or [],
                "role_view": md.get("role_view") or [],
                "source": md.get("source"),
                "entity_ids": md.get("entity_ids") or [],
            })
        return out
    except Exception as ex:
        logger.warning("pinecone query failed (ns=%s): %s", namespace, ex)
        return []

# ----------------- public api -----------------

def retrieve(*, query: str, role: Optional[str], session_id: Optional[str] = None, top_k: int = TOPK_PER_TYPE, tags_any: Optional[List[str]] = None, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """PRD-style retrieval: per-type Pinecone query with role filter and tags_any.
    Returns a flat list of candidates with basic fields required by the pipeline.
    """
    idx = _index()
    if idx is None:
        return []
    vector = embed_text(query)
    if not vector:
        return []

    namespaces = types or ["episodic", "semantic", "procedural"]
    flt = build_metadata_filter(role=role, tags_any=tags_any, extra=None)

    results: List[Dict[str, Any]] = []
    for ns in namespaces:
        results.extend(_query_namespace(idx, vector, ns, top_k, flt))

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return results

def upsert_memory_vector(*, mem_id: str, user_id: Optional[str], type: str, content: str, title: str, tags: List[str], importance: float, created_at_iso: Optional[str], source: str, role_view: Optional[List[str]] = None, entity_ids: Optional[List[str]] = None) -> None:
    """Helper used by ingest to push vectors with required metadata."""
    idx = _index()
    if idx is None or _oai_client is None:
        return
    vec = embed_text(content)
    if not vec:
        return
    metadata = {
        "memory_id": mem_id,
        "type": type,
        "title": title,
        "tags": tags or [],
        "created_at": created_at_iso,
        "role_view": role_view or [],
        "entity_ids": entity_ids or [],
        "source": source,
    }
    try:
        idx.upsert(
            vectors=[{"id": f"mem_{mem_id}", "values": vec, "metadata": metadata}],
            namespace=type,
        )
    except Exception as ex:
        logger.warning("pinecone upsert failed: %s", ex)

__all__ = ["retrieve", "build_metadata_filter", "upsert_memory_vector", "embed_text"]
