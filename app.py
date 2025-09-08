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

# in app.py, after app = FastAPI(...)

def _mount(router_module_name: str):
    try:
        mod = __import__(f"router.{router_module_name}", fromlist=["router"])
        app.include_router(mod.router)
        print(f"[routers] mounted /{router_module_name}")
    except Exception as e:
        print(f"[routers] WARNING: failed to mount '{router_module_name}':", repr(e))

_mount("chat")
_mount("upload")
_mount("ingest_batch")
_mount("memories")        # ← the one we care about
_mount("debug_selftest")  # or your debug router name
_mount("search")


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
