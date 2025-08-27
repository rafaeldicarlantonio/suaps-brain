import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from agent import store, pipeline
from typing import Optional


API_KEY = os.getenv("ACTIONS_API_KEY") or "dev_key"

app = FastAPI(title="SUAPS Agent API")

class ChatInput(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    session_id: str | None = None
    message: str
    history: list[dict] = []
    # Optional chat randomness (0.0–1.2). If None, use backend default.
    temperature: float | None = Field(default=None, description="0.0–1.2")

    @field_validator("temperature")
    @classmethod
    def _clamp_temp(cls, v):
        if v is None:
            return None
        try:
            v = float(v)
        except Exception:
            return None
       # Clamp to a safe range; upstream may still ignore it.
        return max(0.0, min(1.2, v))

@app.get("/healthz")
def healthz():
    return {"ok": True}

def auth(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/chat")
def chat(body: ChatInput, x_api_key: str | None = Header(None)):
    auth(x_api_key)
     user_row = store.ensure_user(body.user_id, body.user_email)
     session_id, answer = pipeline.chat(
         user_id=user_row["id"],
         session_id=body.session_id,
    try:
        # ensure we have a user row and get its UUID
        user_row = store.ensure_user(body.user_id, body.user_email)
        uid = user_row["id"]
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"user resolution error: {ex}")

    try:
        session = {"id": body.session_id} if body.session_id else store.create_session(uid)
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Supabase session error: {ex}")

    try:
        answer = pipeline.run_chat(uid, session["id"], body.history, body.message)
        return {"session_id": session["id"], "answer": answer}
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Chat pipeline error: {ex}")


class MemoryUpsert(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    type: str
    title: str = ""
    content: str
    importance: int = 3
    tags: list[str] = []

@app.post("/memories/upsert")
def memories_upsert(body: MemoryUpsert, x_api_key: str | None = Header(None)):
    auth(x_api_key)
    try:
        user_row = store.ensure_user(body.user_id, body.user_email)
        uid = user_row["id"]
        row = store.upsert_memory(uid, body.type, body.title, body.content, body.importance, body.tags)
        retrieval.upsert_memory_vector(row["id"], uid, body.type, body.content, body.title, body.tags, body.importance)
        return {"id": row["id"]}
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"Upsert memory error: {ex}")



from agent.ingest import distill_chunk

class IngestItem(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    text: str
    type: str = "semantic"
    tags: list[str] = []

@app.post("/ingest/batch")
def ingest_batch(body: dict, x_api_key: str | None = Header(None)):
    auth(x_api_key)
    try:
        items = body.get("items", [])
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        out = []
        for it in items:
            user_row = store.ensure_user(it.get("user_id"), it.get("user_email"))
            uid = user_row["id"]
            t = (it.get("text") or "").strip()
            if not t:
                continue
            if it.get("type") == "semantic":
                ids = distill_chunk(user_id=uid, raw_text=t, base_tags=it.get("tags",[]), make_qa=True)
                out.extend(ids)
            else:
                row = store.upsert_memory(uid, "episodic", "", t, 4, it.get("tags",[]))
                retrieval.upsert_memory_vector(row["id"], uid, "episodic", t, "", it.get("tags",[]), 4)
                out.append(row["id"])
        return {"created_ids": out}
    except Exception as ex:
        # echo exact error text to client for debugging
        raise HTTPException(status_code=502, detail=f"Ingest error: {ex}")




from vendors.openai_client import client as _openai
from vendors.pinecone_client import pc as _pc, INDEX as _pine_index
from vendors.supabase_client import supabase as _supabase

@app.get("/debug/selftest")
def selftest(x_api_key: str | None = Header(None)):
    auth(x_api_key)
    out = {"openai": None, "pinecone": None, "supabase": None}

    # OpenAI
    try:
        e = _openai.embeddings.create(model=os.getenv("OPENAI_EMBED_MODEL","text-embedding-3-small"), input="ping")
        out["openai"] = {"ok": True, "dim": len(e.data[0].embedding)}
    except Exception as ex:
        out["openai"] = {"ok": False, "error": str(ex)}

    # Pinecone
    try:
        idx_name = os.getenv("PINECONE_INDEX","memories")
        names = [i["name"] for i in _pc.list_indexes()]
        if idx_name not in names:
            out["pinecone"] = {"ok": False, "error": f"Index '{idx_name}' not found. Existing: {names}"}
        else:
            # light query (won't fail if empty, but vector required)
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
