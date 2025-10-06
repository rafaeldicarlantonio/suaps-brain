# app.py — mounts routers, exposes health, and surfaces mount failures

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load config once at startup. Fail loudly if env is broken.
from config import (
    load_config,
    MAX_CONTEXT_TOKENS,
    TOPK_PER_TYPE,
    RECENCY_HALFLIFE_DAYS,
    RECENCY_FLOOR,
)

CFG = load_config()

app = FastAPI(
    title="SUAPS Brain",
    version="0.1.0",
    description="RAG-ish brain that now actually tells you when things are broken."
)

# CORS — permissive for now; lock down later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track router mount failures so 404s aren’t mysteries.
_router_failures = []
_mounted = []

def _mount(router_module_name: str):
    try:
        mod = __import__(f"router.{router_module_name}", fromlist=["router"])
        app.include_router(mod.router)
        _mounted.append(router_module_name)
        print(f"[routers] mounted /{router_module_name}")
    except Exception as e:
        msg = repr(e)
        _router_failures.append({"router": router_module_name, "error": msg})
        print(f"[routers] WARNING: failed to mount '{router_module_name}': {msg}")

# Mount all known routers. If any explode at import-time, we’ll see it in /debug/routers.
_mount("chat")
_mount("upload")
_mount("ingest")
_mount("memories")
_mount("debug_selftest")
_mount("search")
_mount("entities")

@app.get("/debug/routers")
def debug_routers():
    """
    Shows which routers mounted successfully and which face-planted at import time.
    Use this the next time /search returns 404 and you swear it exists.
    """
    return {
        "mounted": _mounted,
        "failures": _router_failures,
    }

@app.get("/healthz")
async def healthz():
    """
    Cheap liveness probe with vendor init sanity.
    """
    status = {
        "status": "ok",
        "supabase": False,
        "pinecone": False,
        "openai": False,
        # helpful config echoes (not secrets)
        "cfg": {
            "embed_model": CFG.get("EMBED_MODEL"),
            "memories_text_column": CFG.get("MEMORIES_TEXT_COLUMN"),
            "embed_dim": CFG.get("EMBED_DIM"),
            "pinecone_index": CFG.get("PINECONE_INDEX"),
        },
    }

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
    status["openai"] = bool(os.getenv("OPENAI_API_KEY"))

    return status
