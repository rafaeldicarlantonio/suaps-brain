import os
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Optional project modules (fail-safe imports)
try:
    from agent import store, retrieval
    from agent.ingest import distill_chunk
except Exception:
    store = None
    retrieval = None
    distill_chunk = None

# Routers (comment out if a router is missing)
try:
    from router.upload import router as upload_router
except Exception:
    upload_router = None
try:
    from router.chat import router as chat_router
except Exception:
    chat_router = None
try:
    from router.debug import router as debug_router
except Exception:
    debug_router = None

# Create the app ONCE
app = FastAPI(title="SUAPS Brain API", version="1.0.0")

# Register routers AFTER app is created (only if they exist)
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
API_KEY = os.getenv("ACTIONS_API_KEY") or "dev_key"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
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
    """
    Normalize ANY input into a real UUID from `users`.
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
    return {"ok": True, "auth_disabled": DISABLE_AUTH}

@app.get("/whoami")
def whoami(user_id: Optional[str] = None, user_email: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    u = _resolve_user(user_id, user_email)
    return {"authorized": True, "user": {"id": u["id"], "email": u.get("email")}}

# --------------------------------------------------------------------
# Memories Upsert
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
        return {"id": row["id"]}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Upsert memory error: {ex}")

# --------------------------------------------------------------------
# Ingest Batch
# --------------------------------------------------------------------
class IngestItem(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    text: str
    type: str = "semantic"  # or "episodic"
    tags: List[str] = []

class IngestBatch(BaseModel):
    items: List[IngestItem]

@app.post("/ingest/batch")
def ingest_batch(body: IngestBatch, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    if store is None or retrieval is None:
        raise HTTPException(status_code=500, detail="store/retrieval modules unavailable")
    try:
        out: List[str] = []
        for it in body.items:
            # HARD STOP: ignore incoming user_id; always resolve to a UUID
            email_or_alias = it.user_email if it.user_email else it.user_id
            user_row = _resolve_user(None, email_or_alias)
            uid = user_row["id"]
            t = (it.text or "").strip()
            if not t:
                continue
            if it.type == "semantic":
                if distill_chunk is None:
                    raise HTTPException(status_code=500, detail="distill_chunk unavailable")
                ids = distill_chunk(user_id=uid, raw_text=t, base_tags=it.tags or [], make_qa=True)
                out.extend(ids or [])
            else:
                row = store.upsert_memory(uid, "episodic", "", t, 4, it.tags or [])
                retrieval.upsert_memory_vector(
                    mem_id=row["id"],
                    user_id=uid,
                    type="episodic",       # NOTE: kwarg is 'type', not 'type_'
                    content=t,
                    title="",
                    tags=it.tags or [],
                    importance=4,
                    created_at_iso=None,
                    source="ingest",
                    role_view=None,
                    entity_ids=[],
                )
                out.append(row["id"])
        return {"created_ids": out}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Ingest error: {ex}")


# --------------------------------------------------------------------
# Debug / Selftest
# --------------------------------------------------------------------
try:
    from vendors.openai_client import client as _openai
except Exception:
    _openai = None
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
        if _openai is None:
            raise RuntimeError("openai client missing")
        e = _openai.embeddings.create(model=OPENAI_EMBED_MODEL, input="ping")
        out["openai"] = {"ok": True, "dim": len(e.data[0].embedding)}
    except Exception as ex:
        out["openai"] = {"ok": False, "error": str(ex)}

    try:
        if _pc is None:
            raise RuntimeError("pinecone client missing")
        idx_name = os.getenv("PINECONE_INDEX", "memories")
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
