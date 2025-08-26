import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from agent import store, pipeline

API_KEY = os.getenv("ACTIONS_API_KEY") or "dev_key"

app = FastAPI(title="SUAPS Agent API")

class ChatInput(BaseModel):
    user_id: str
    session_id: str | None = None
    message: str
    history: list[dict] = []  # [{'role':'user'|'assistant','content': '...'}]

@app.get("/healthz")
def healthz():
    return {"ok": True}

def auth(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/chat")
def chat(body: ChatInput, x_api_key: str | None = Header(None)):
    auth(x_api_key)
    session = {"id": body.session_id} if body.session_id else store.create_session(body.user_id)
    answer = pipeline.run_chat(body.user_id, session["id"], body.history, body.message)
    return {"session_id": session["id"], "answer": answer}

class MemoryUpsert(BaseModel):
    user_id: str
    type: str
    title: str = ""
    content: str
    importance: int = 3
    tags: list[str] = []

@app.post("/memories/upsert")
def memories_upsert(body: MemoryUpsert, x_api_key: str | None = Header(None)):
    auth(x_api_key)
    from agent import retrieval, store
    row = store.upsert_memory(body.user_id, body.type, body.title, body.content, body.importance, body.tags)
    retrieval.upsert_memory_vector(row["id"], body.user_id, body.type, body.content, body.title, body.tags, body.importance)
    return {"id": row["id"]}

from agent.ingest import distill_chunk

class IngestItem(BaseModel):
    user_id: str
    text: str
    type: str = "semantic"
    tags: list[str] = []

@app.post("/ingest/batch")
def ingest_batch(body: dict, x_api_key: str | None = Header(None)):
    auth(x_api_key)
    items = body.get("items", [])
    out = []
    for it in items:
        if it.get("type") == "semantic":
            ids = distill_chunk(user_id=it["user_id"], raw_text=it["text"], base_tags=it.get("tags",[]), make_qa=True)
            out.extend(ids)
        else:
            row = store.upsert_memory(it["user_id"], "episodic", "", it["text"], 4, it.get("tags",[]))
            retrieval.upsert_memory_vector(row["id"], it["user_id"], "episodic", it["text"], "", it.get("tags",[]), 4)
            out.append(row["id"])
    return {"created_ids": out}
