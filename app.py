import os
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Routers
# - ingest router is a MODULE exposing `router = APIRouter()`
# - upload/chat/debug import the APIRouter object directly as `router`
try:
    from router.ingest_batch import router as ingest_router  # module
except Exception:
    ingest_router = None

try:
    from router.upload import router as upload_router  # APIRouter
except Exception:
    upload_router = None

try:
    from router.chat import router as chat_router      # APIRouter
except Exception:
    chat_router = None

try:
    from router.debug import router as debug_router    # APIRouter
except Exception:
    debug_router = None

# Optional project modules (fail-safe imports)
try:
    from agent import store, retrieval
    from agent.ingest import distill_chunk
except Exception:
    store = None
    retrieval = None
    distill_chunk = None

# Create the app ONCE
app = FastAPI(title="SUAPS Brain API", version="1.0.0")

# --- Include routers (fixes previous NameError: use app.include_router) ---
if ingest_router is not None and hasattr(ingest_router, "router"):
    # This provides POST /ingest/batch via router/ingest_batch.py
    app.include_router(ingest_router)

if upload_router is not None:
    app.include_router(upload_router)

if chat_router is not None:
    app.include_router(chat_router)

if debug_router is not None:
    app.include_router(debug_router)

# Optional: avoid 404 on root
@app.get("/")
def root():
    return {"ok": True, "service": "suaps-brain"}

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
API_KEY = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY") or "dev_key"
DISABLE_AUTH = os.getenv("DISABLE_AUTH", "false").lower() == "true"

# --------------------------------------------------------------------
# Security (bypass when DISABLE_AUTH=true)
# --------------------------------------------------------------------
def auth(x_api_key: Optional[str]):
    if DISABLE_AUTH:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized: missing X-API-Key header")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid X-API-Key value")

def _is_uuid(s: Optional[str]) -> bool:
    if not s:
        return False
    try:
        uuid.UUID(str(s))
        return True
    except Exception:
        return False

def _resolve_user(user_id: Optional[str], user_email: Optional[str]) -> Dict:
    """Normalize ANY input into a real UUID from `users`.

    Priority:

      1) user_email provided -> upsert/select by email

      2) user_id is UUID     -> select by id

      3) user_id not UUID    -> treat as email/alias

      4) none provided       -> anonymous

    """
    if store is None:
        raise HTTPException(status_code=500, detail="store module unavailable")
    if not user_email and user_id and not _is_uuid(user_id):
        user_email = user_id
        user_id = None
    if not user_id and not user_email:
        user_email = "anonymous@suaps.local"
    return store.ensure_user(user_id, user_email)

# --------------------------------------------------------------------
# Health & Whoami
# --------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    """PRD §5.5: return explicit connectivity booleans."""
    out = {"status": "ok", "openai": False, "pinecone": False, "supabase": False}
    # OpenAI
    try:
        from vendors.openai_client import client, EMBED_MODEL
        client.embeddings.create(model=EMBED_MODEL, input="ping")
        out["openai"] = True
    except Exception:
        out["openai"] = False
    # Pinecone
    try:
        from vendors.pinecone_client import pc
        idxs = pc.list_indexes()
        out["pinecone"] = True if idxs is not None else False
    except Exception:
        out["pinecone"] = False
    # Supabase
    try:
        from vendors.supabase_client import supabase
        supabase.table("memories").select("id").limit(1).execute()
        out["supabase"] = True
    except Exception:
        out["supabase"] = False
    return out


@app.get("/whoami")
def whoami(user_id: Optional[str] = None, user_email: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    u = _resolve_user(user_id, user_email)
    return {"authorized": True, "user": {"id": u["id"], "email": u.get("email")}}

# --------------------------------------------------------------------
# Memories Upsert (kept in app for now)
# --------------------------------------------------------------------
class MemoryUpsert(BaseModel):
    # IMPORTANT: we will IGNORE user_id below and always use resolved UUID
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    type: str
    title: str = ""
    content: str
    importance: int = 3
    tags: List[str] = []

@app.post("/memories/upsert")
def memories_upsert(body: MemoryUpsert, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    if store is None or retrieval is None:
        raise HTTPException(status_code=500, detail="store/retrieval modules unavailable")
    try:
        # HARD STOP: ignore incoming user_id; always resolve to a UUID
        user_row = _resolve_user(None, body.user_email if body.user_email else body.user_id)
        uid = user_row["id"]
        row = store.upsert_memory(uid, body.type, body.title, body.content, body.importance, body.tags)
        retrieval.upsert_memory_vector(
            mem_id=row["id"],
            user_id=uid,
            type=body.type,               # NOTE: kwarg is 'type', not 'type_'
            content=body.content,
            title=body.title,
            tags=body.tags,
            importance=body.importance,
            created_at_iso=None,
            source="chat",
            role_view=None,
            entity_ids=[],
        )
        store.update_memory_embedding_id(row["id"], f"mem_{row['id']}")
        return {"id": row["id"]}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Upsert memory error: {ex}")

# --------------------------------------------------------------------
# IMPORTANT: Do NOT define /ingest/batch here to avoid conflicting with router/ingest_batch.py
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Debug / Selftest
# --------------------------------------------------------------------
try:
    from vendors.openai_client import client as _openai, EMBED_MODEL as _EMBED_MODEL
except Exception:
    _openai = None
    _EMBED_MODEL = None
try:
    from vendors.pinecone_client import pc as _pc
except Exception:
    _pc = None
try:
    from vendors.supabase_client import supabase as _supabase
except Exception:
    _supabase = None

@app.get("/debug/selftest")
def selftest(x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    out = {"openai": None, "pinecone": None, "supabase": None, "auth_disabled": DISABLE_AUTH}

    try:
        if _openai is None or _EMBED_MODEL is None:
            raise RuntimeError("openai client or EMBED_MODEL missing")
        e = _openai.embeddings.create(model=_EMBED_MODEL, input="ping")
        out["openai"] = {"ok": True, "dim": len(e.data[0].embedding)}
    except Exception as ex:
        out["openai"] = {"ok": False, "error": str(ex)}

    try:
        if _pc is None:
            raise RuntimeError("pinecone client missing")
        idx_name = os.getenv("PINECONE_INDEX", "uap-kb")
        names = [i["name"] for i in _pc.list_indexes()]
        if idx_name not in names:
            out["pinecone"] = {"ok": False, "error": f"Index '{idx_name}' not found. Existing: {names}"}
        else:
            out["pinecone"] = {"ok": True, "index": idx_name}
    except Exception as ex:
        out["pinecone"] = {"ok": False, "error": str(ex)}

    try:
        if _supabase is None:
            raise RuntimeError("supabase client missing")
        r = _supabase.table("memories").select("id").limit(1).execute()
        out["supabase"] = {"ok": True, "count": len(r.data or [])}
    except Exception as ex:
        out["supabase"] = {"ok": False, "error": str(ex)}

    return out
