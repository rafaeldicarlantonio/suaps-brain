from __future__ import annotations

# ---------- app.py (Phase 0 safe baseline) ----------

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# (A) Config guard — raises at startup if critical env vars are missing
# Put this in app/config.py exactly as given in Phase 0.
try:
    from app.config import (
        MAX_CONTEXT_TOKENS,
        TOPK_PER_TYPE,
        RECENCY_HALFLIFE_DAYS,
        RECENCY_FLOOR,
    )
except Exception as e:
    # Fail fast with a clear error if envs are missing
    raise

app = FastAPI(title="SUAPS Brain", version="0.0.1-phase0")

# CORS (restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (B) Include routers — keep them even if they return simple placeholders for now
# If your router modules are under app/router/, these imports work as-is.
try:
    from app.router import chat, upload, ingest_batch, memories, debug_selftest
    app.include_router(chat.router, prefix="")
    app.include_router(upload.router, prefix="")
    app.include_router(ingest_batch.router, prefix="")
    app.include_router(memories.router, prefix="")
    app.include_router(debug_selftest.router, prefix="")
except Exception:
    # Don't crash app startup if a router module is incomplete in Phase 0.
    # We'll fix the modules in the next phases.
    pass


# (C) Safe health check — reports booleans without throwing 500s
@app.get("/healthz")
async def healthz():
    status = {"status": "ok", "supabase": False, "pinecone": False, "openai": False}

    # Supabase ping (best-effort)
    try:
        # Adjust import to your actual helper
        from app.vendors.supabase_client import get_client  # e.g., returns a supabase client
        sb = get_client()
        # Lightweight check: run a trivial RPC/list call or just ensure object exists
        status["supabase"] = bool(sb)
    except Exception:
        status["supabase"] = False

    # Pinecone ping (best-effort)
    try:
        # Adjust import to your actual helper
        from app.vendors.pinecone_client import get_index  # e.g., returns a pinecone Index
        idx = get_index()
        # Avoid listing index stats here in Phase 0 to keep it fast; existence is enough
        status["pinecone"] = bool(idx)
    except Exception:
        status["pinecone"] = False

    # OpenAI ping (best-effort)
    try:
        # Adjust if you centralize this elsewhere
        import openai  # make sure OPENAI_API_KEY is set
        status["openai"] = bool(os.getenv("OPENAI_API_KEY"))
    except Exception:
        status["openai"] = False

    return status
