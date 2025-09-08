# router/debug_selftest.py
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from vendors.supabase_client import get_client

router = APIRouter()


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


@router.get("/debug/selftest")
def debug_selftest(x_api_key: Optional[str] = Header(None)):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    out = {
        "supabase": False,
        "openai": False,
        "embedding": False,
        "pinecone": False,
        "pinecone_roundtrip": False,
        "latency_ms": {},
    }

    # --- Supabase: simple select on memories ---
    t0 = time.time()
    try:
        sb = get_client()
        sb.table("memories").select("id").limit(1).execute()
        out["supabase"] = True
    except Exception as e:
        out["supabase_error"] = str(e)
    out["latency_ms"]["supabase"] = _ms(t0)

    # --- OpenAI: create a tiny embedding we can reuse for Pinecone test ---
    embed_vec = None
    t0 = time.time()
    try:
        from openai import OpenAI

        oai = OpenAI()
        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        er = oai.embeddings.create(model=model, input="selftest probe")
        embed_vec = er.data[0].embedding
        out["openai"] = True
        out["embedding"] = bool(embed_vec)
    except Exception as e:
        out["openai_error"] = str(e)
    out["latency_ms"]["openai"] = _ms(t0)

    # --- Pinecone: connectivity ---
    t0 = time.time()
    try:
        from vendors.pinecone_client import get_index

        index = get_index()
        out["pinecone"] = True
    except Exception as e:
        out["pinecone_error"] = str(e)
        out["latency_ms"]["pinecone"] = _ms(t0)
        return out  # can't proceed to round-trip without an index
    out["latency_ms"]["pinecone"] = _ms(t0)

    # --- Pinecone: round-trip (upsert -> fetch -> delete) ---
    if embed_vec:
        ns = os.getenv("DEBUG_NAMESPACE", "debug-selftest")
        vid = f"selftest_{uuid.uuid4().hex[:12]}"
        meta = {"reason": "selftest", "created_at": int(time.time())}

        t0 = time.time()
        try:
            # Upsert
            index.upsert(
                vectors=[{"id": vid, "values": embed_vec, "metadata": meta}],
                namespace=ns,
            )

            # Fetch (SDKs differ: dict vs FetchResponse)
            fetched = index.fetch(ids=[vid], namespace=ns)
            if isinstance(fetched, dict):
                vectors = fetched.get("vectors", {}) or {}
            else:
                vectors = getattr(fetched, "vectors", {}) or {}

            if isinstance(vectors, dict) and vid in vectors:
                out["pinecone_roundtrip"] = True

            # Cleanup (best-effort)
            try:
                index.delete(ids=[vid], namespace=ns)
            except Exception:
                pass

            out["latency_ms"]["pinecone_roundtrip"] = _ms(t0)
        except Exception as e:
            out["pinecone_roundtrip_error"] = str(e)
            out["latency_ms"]["pinecone_roundtrip"] = _ms(t0)

    return out


@router.get("/debug/memories")
def debug_memories(
    limit: int = Query(20, ge=1, le=100),
    type: Optional[str] = Query(None, regex="^(episodic|semantic|procedural)$"),
    x_api_key: Optional[str] = Header(None),
):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    sb = get_client()
    q = (
        sb.table("memories")
        .select("id,type,title,tags,created_at,embedding_id")
        .order("created_at", desc=True)
        .limit(limit)
    )
    if type:
        q = q.eq("type", type)
    res = q.execute()
    data = res.data if hasattr(res, "data") else res.get("data")
    return {"items": data or []}
