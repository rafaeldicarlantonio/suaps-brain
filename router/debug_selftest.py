from __future__ import annotations
import time
from fastapi import APIRouter
from vendors.openai_client import client, EMBED_MODEL
from vendors.pinecone_client import get_index

router = APIRouter(tags=["debug"])

@router.get("/debug/selftest")
def selftest():
    t0=time.time()
    e = client.embeddings.create(model=EMBED_MODEL, input="ping").data[0].embedding
    pine_t = time.time()
    idx = get_index()
    try:
        idx.query(vector=e, top_k=1, namespace="semantic")
        pine_ok = True
    except Exception as ex:
        pine_ok = False
    return {
        "openai_ms": int((pine_t - t0)*1000),
        "pinecone_ok": pine_ok,
        "latency_ms": int((time.time()-t0)*1000),
    }
