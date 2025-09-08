# app.py — Phase 0 safe baseline (root-relative imports)

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# (A) Config guard – now imported from root-level config.py
from config import (
    MAX_CONTEXT_TOKENS,
    TOPK_PER_TYPE,
    RECENCY_HALFLIFE_DAYS,
    RECENCY_FLOOR,
)

app = FastAPI(title="SUAPS Brain", version="0.0.1-phase0")

# (B) CORS – permissive for Phase 0; we’ll tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (C) Routers – import from top-level 'router/' package if that’s your structure
try:
    from router import chat, upload, ingest_batch, memories, debug_selftest
    app.include_router(chat.router)
    app.include_router(upload.router)
    app.include_router(ingest_batch.router)
    app.include_router(memories.router)
    app.include_router(debug_selftest.router)
except Exception as e:
    # Keep server booting in Phase 0, even if a router is incomplete
    print("Router load warning:", repr(e))

# (D) Health – use root-level 'vendors/' helpers
@app.get("/healthz")
async def healthz():
    status = {"status": "ok", "supabase": False, "pinecone": False, "openai": False}

    # Supabase
    try:
        from vendors.supabase_client import get_client
        status["supabase"] = bool(get_client())
    except Exception:
        status["supabase"] = False

    # Pinecone
    try:
        from vendors.pinecone_client import get_index
        status["pinecone"] = bool(get_index())
    except Exception:
        status["pinecone"] = False

    # OpenAI
    try:
        status["openai"] = bool(os.getenv("OPENAI_API_KEY"))
    except Exception:
        status["openai"] = False

    return status
