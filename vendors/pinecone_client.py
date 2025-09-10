# vendors/pinecone_client.py
import os
from types import SimpleNamespace
from pinecone import Pinecone, ServerlessSpec

_pc_singleton = None
_index = None

def get_index():
    global _pc_singleton, _index
    if _index: 
        return _index
    _pc_singleton = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    name = os.getenv("PINECONE_INDEX", "uap-kb")
    _index = _pc_singleton.Index(name)
    return _index

def safe_query(index, **kwargs):
    """
    Always return an object with .matches (list of Ns with .id, .score, .metadata dict).
    Works whether the SDK returns objects or dict-like.
    """
    resp = index.query(**kwargs)
    # normalize
    data = resp.to_dict() if hasattr(resp, "to_dict") else (resp if isinstance(resp, dict) else None)
    if data:
        out = []
        for m in data.get("matches", []):
            out.append(SimpleNamespace(
                id=m.get("id"),
                score=m.get("score"),
                metadata=m.get("metadata") or {}
            ))
        return SimpleNamespace(matches=out)
    # fallback to attribute style
    if hasattr(resp, "matches"):
        out = []
        for m in (resp.matches or []):
            md = getattr(m, "metadata", {}) or {}
            out.append(SimpleNamespace(id=getattr(m, "id", None), score=getattr(m, "score", 0.0), metadata=md))
        return SimpleNamespace(matches=out)
    # last resort
    return SimpleNamespace(matches=[])
