from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Instantiate app FIRST (fixes early include bugs)
app = FastAPI(title="SUAPS Brain", version=os.getenv("APP_VERSION","0.1.0"))

# Basic CORS for Actions/Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from router.chat import router as chat_router
from router.upload import router as upload_router
from router.ingest_batch import router as ingest_router
from router.debug import router as debug_router
from router.debug_selftest import router as selftest_router
from router.memories import router as memories_router
from schemas.api import HealthzResponse

app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(ingest_router)
app.include_router(debug_router)
app.include_router(selftest_router)
app.include_router(memories_router)

# Healthz (PRD §5.5)
@app.get("/healthz", response_model=HealthzResponse)
def healthz():
    out = {"status":"ok"}
    # OpenAI
    try:
        from vendors.openai_client import client, CHAT_MODEL
        client.models.list()
        out["openai"] = {"ok": True, "chat_model": CHAT_MODEL}
    except Exception as ex:
        out["openai"] = {"ok": False, "error": str(ex)}
    # Pinecone
    try:
        from vendors.pinecone_client import get_index, INDEX_NAME
        idx = get_index()
        _ = getattr(idx, "describe_index_stats")(None) if hasattr(idx, "describe_index_stats") else {}
        out["pinecone"] = {"ok": True, "index": INDEX_NAME}
    except Exception as ex:
        out["pinecone"] = {"ok": False, "error": str(ex)}
    # Supabase
    try:
        from vendors.supabase_client import supabase
        r = supabase.table("memories").select("id").limit(1).execute()
        out["supabase"] = {"ok": True, "count": len(r.data or [])}
    except Exception as ex:
        out["supabase"] = {"ok": False, "error": str(ex)}
    return out
