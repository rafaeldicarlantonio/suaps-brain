# router/debug_selftest.py
from fastapi import APIRouter
import time
from vendors.openai_client import client, EMBED_MODEL
from vendors.pinecone_client import INDEX

router = APIRouter()

@router.get("/debug/selftest")
def selftest():
    t0=time.time()
    try:
        e = client.embeddings.create(model=EMBED_MODEL, input="ping").data[0].embedding
        pine_t = time.time()
        INDEX.query(vector=e, top_k=1, namespace="semantic")
        pine_lat = int((time.time()-pine_t)*1000)
        return {"ok":True,"latency_ms":int((time.time()-t0)*1000),"pinecone_ms":pine_lat}
    except Exception as ex:
        return {"ok":False,"error":str(ex),"latency_ms":int((time.time()-t0)*1000)}
