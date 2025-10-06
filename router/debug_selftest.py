# router/debug_selftest.py
import os, time
from typing import Optional, Dict, Any
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index, safe_query

router = APIRouter()

class SelftestOut(BaseModel):
    supabase: bool
    openai: bool
    embedding: bool
    pinecone: bool
    pinecone_roundtrip: bool
    latency_ms: Dict[str, int]

def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

@router.get("/debug/selftest", response_model=SelftestOut)
def debug_selftest_get(x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    lat = {"supabase": 0, "openai": 0, "pinecone": 0, "pinecone_roundtrip": 0}
    ok_sb = ok_oai = ok_emb = ok_pc = ok_round = False

    # Supabase ping
    try:
        t0 = time.time()
        sb = get_client()
        sb.table("memories").select("id").limit(1).execute()
        lat["supabase"] = int((time.time() - t0) * 1000)
        ok_sb = True
    except Exception:
        ok_sb = False

    # OpenAI + embed
    try:
        from openai import OpenAI
        oai = OpenAI()
        t0 = time.time()
        er = oai.embeddings.create(model=os.getenv("EMBED_MODEL","text-embedding-3-small"), input="selftest")
        _ = er.data[0].embedding
        lat["openai"] = int((time.time() - t0) * 1000)
        ok_oai = ok_emb = True
    except Exception:
        ok_oai = ok_emb = False

    # Pinecone index
    try:
        t0 = time.time()
        index = get_index()
        lat["pinecone"] = int((time.time() - t0) * 1000)
        ok_pc = True
    except Exception:
        ok_pc = False

    # Pinecone roundtrip
    if ok_pc and ok_emb:
        try:
            t0 = time.time()
            vec = er.data[0].embedding
            qr = safe_query(index, vector=vec, top_k=1, include_metadata=True, namespace="semantic")
            _ = list(qr.matches)  # normalize iterable
            lat["pinecone_roundtrip"] = int((time.time() - t0) * 1000)
            ok_round = True
        except Exception:
            ok_round = False

    return {
        "supabase": ok_sb,
        "openai": ok_oai,
        "embedding": ok_emb,
        "pinecone": ok_pc,
        "pinecone_roundtrip": ok_round,
        "latency_ms": lat,
    }
def _embedding_dim(client, model, dim_override=None):
    # create a tiny embedding to measure the real output size
    kwargs = {"model": model, "input": "dimension check"}
    if dim_override:
        kwargs["dimensions"] = dim_override
    vec = client.embeddings.create(**kwargs).data[0].embedding
    return len(vec)

@router.get("/debug/selftest")
def selftest():
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    dim_override = os.getenv("EMBED_DIM")
    dim = _embedding_dim(client, model, int(dim_override) if dim_override else None)

    index = get_index()
    # assuming your vendor has index.describe() or you stored it during creation
    index_dim = index._description.dimension  # adapt to your client

    if dim != index_dim:
        return {"ok": False, "error": f"Embedding dim {dim} != Pinecone index dim {index_dim}"}
    return {"ok": True, "embedding_dim": dim, "index_dim": index_dim}
