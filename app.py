# app.py
import os
import uuid
from typing import Optional, List, Dict

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from agent import store, pipeline, retrieval
from agent.ingest import distill_chunk

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
API_KEY = os.getenv("ACTIONS_API_KEY") or "dev_key"
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

app = FastAPI(title="SUAPS Agent API")

# --------------------------------------------------------------------
# Security
# --------------------------------------------------------------------
def auth(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
    Allow three forms:
      - user_email provided  -> upsert by email
      - user_id is UUID      -> resolve by UUID
      - user_id is non-UUID  -> treat as email/alias and upsert by that
    """
    if not user_email and user_id and not _is_uuid(user_id):
        user_email = user_id
        user_id = None
    return store.ensure_user(user_id, user_email)

# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------
class ChatInput(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    session_id: Optional[str] = None
    message: str
    history: List[Dict] = []
    temperature: Optional[float] = Field(None, description="0.0–1.2 (optional)")

    @field_validator("temperature")
    @classmethod
    def clamp_temp(cls, v):
        if v is None:
            return None
        try:
            v = float(v)
        except Exception:
            return None
        return max(0.0, min(1.2, v))

class MemoryUpsert(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    type: str
    title: str = ""
    content: str
    importance: int = 3
    tags: List[str] = []

class IngestItem(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    text: str
    type: str = "semantic"  # or "episodic"
    tags: List[str] = []

class IngestBatch(BaseModel):
    items: List[IngestItem]

# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# --------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------
@app.post("/chat")
def chat(body: ChatInput, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    try:
        user_row = _resolve_user(body.user_id, body.user_email)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"user resolution error: {ex}")

    try:
        session_id, answer = pipeline.chat(
            user_id=user_row["id"],
            session_id=body.session_id,
            message=body.message,
            history=body.history,
            temperature=body.temperature,  # may be None; pipeline decides default/omit
        )
        return {"session_id": session_id, "answer": answer}
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"chat pipeline error: {ex}")

# --------------------------------------------------------------------
# Memories Upsert
# --------------------------------------------------------------------
@app.post("/memories/upsert")
def memories_upsert(body: MemoryUpsert, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    try:
        user_row = _resolve_user(body.user_id, body.user_email)
        uid = user_row["id"]
        row = store.upsert_memory(
            uid, body.type, body.title, body.content, body.importance, body.tags
        )
        # keep vector store in sync
        retrieval.upsert_memory_vector(
            mem_id=row["id"],
            user_id=uid,
            type_=body.type,
            content=body.content,
            title=body.title,
            tags=body.tags,
            importance=body.importance,
        )
        return {"id": row["id"]}
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Upsert memory error: {ex}")

# --------------------------------------------------------------------
# Ingest Batch
# --------------------------------------------------------------------
@app.post("/ingest/batch")
def ingest_batch(body: IngestBatch, x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    try:
        out: List[str] = []
        for it in body.items:
            user_row = _resolve_user(it.user_id, it.user_email)
            uid = user_row["id"]
            t = (it.text or "").strip()
            if not t:
                continue
            if it.type == "semantic":
                ids = distill_chunk(
                    user_id=uid, raw_text=t, base_tags=it.tags or [], make_qa=True
                )
                out.extend(ids)
            else:
                row = store.upsert_memory(uid, "episodic", "", t, 4, it.tags or [])
                retrieval.upsert_memory_vector(
                    mem_id=row["id"],
                    user_id=uid,
                    type_="episodic",
                    content=t,
                    title="",
                    tags=it.tags or [],
                    importance=4,
                )
                out.append(row["id"])
        return {"created_ids": out}
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Ingest error: {ex}")

# --------------------------------------------------------------------
# Debug / Selftest
# --------------------------------------------------------------------
from vendors.openai_client import client as _openai
from vendors.pinecone_client import pc as _pc
from vendors.supabase_client import supabase as _supabase

@app.get("/debug/selftest")
def selftest(x_api_key: Optional[str] = Header(None)):
    auth(x_api_key)
    out = {"openai": None, "pinecone": None, "supabase": None}

    # OpenAI
    try:
        e = _openai.embeddings.create(model=OPENAI_EMBED_MODEL, input="ping")
        out["openai"] = {"ok": True, "dim": len(e.data[0].embedding)}
    except Exception as ex:
        out["openai"] = {"ok": False, "error": str(ex)}

    # Pinecone
    try:
        idx_name = os.getenv("PINECONE_INDEX", "memories")
        names = [i["name"] for i in _pc.list_indexes()]
        if idx_name not in names:
            out["pinecone"] = {
                "ok": False,
                "error": f"Index '{idx_name}' not found. Existing: {names}",
            }
        else:
            out["pinecone"] = {"ok": True, "index": idx_name}
    except Exception as ex:
        out["pinecone"] = {"ok": False, "error": str(ex)}

    # Supabase
    try:
        r = _supabase.table("memories").select("id").limit(1).execute()
        out["supabase"] = {"ok": True, "count": len(r.data or [])}
    except Exception as ex:
        out["supabase"] = {"ok": False, "error": str(ex)}

    return out
