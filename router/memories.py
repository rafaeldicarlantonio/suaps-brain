from typing import Optional, List, Any, Dict, Literal
import os, re, hashlib, datetime
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, root_validator, validator

router = APIRouter()
_ALLOWED_TYPES = {"episodic", "semantic", "procedural"}

# ---------- Pydantic payload (accepts extras but ignores them) ----------

class MemoryUpsertIn(BaseModel):
    type: Literal["episodic", "semantic", "procedural"]
    title: Optional[str] = None

    # Accept 'text' normally; also tolerate 'value' by coalescing it into text.
    text: Optional[str] = Field(default=None)
    value: Optional[str] = Field(default=None)

    tags: List[str] = Field(default_factory=list)
    role_view: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(default="chat")

    @root_validator(pre=True)
    def coalesce_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Prefer explicit 'text'; else take 'value'
        raw = values.get("text") or values.get("value")
        if not raw or not str(raw).strip():
            raise ValueError("Either 'text' or 'value' must be provided and non-empty.")
        values["text"] = str(raw)
        return values

    @validator("tags", "role_view", pre=True)
    def listify(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    @validator("source", pre=True)
    def normalize_source(cls, v):
        s = (v or "chat").strip().lower()
        return s if s in {"upload", "ingest", "chat", "wiki"} else "chat"

    class Config:
        extra = "ignore"  # ← silently ignore unknown keys like role_metadata

# ---------- Helpers ----------

def _norm_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s or "")
    s = re.sub(r"\s+\n", "\n", s).strip()
    return s

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def _resp_data(resp) -> List[Dict[str, Any]]:
    if resp is None: return []
    if hasattr(resp, "data"):  # supabase-py v2
        return resp.data or []
    if isinstance(resp, dict):
        return resp.get("data") or []
    return []

# ---------- Core logic shared by all route aliases ----------

async def _memories_upsert_core(request: Request, x_api_key: Optional[str]) -> Dict[str, Any]:
    # Optional API key
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Parse & validate body
    try:
        data = await request.json()
        payload = MemoryUpsertIn.parse_obj(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    mem_type = payload.type
    title = (payload.title or "").strip()
    text = _norm_text(payload.text)
    tags = payload.tags
    role_view = payload.role_view
    source = payload.source

    dedupe_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Supabase client
    try:
        from vendors.supabase_client import get_client
        sb = get_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase client not available: {e}")

    # 1) Check duplicate
    try:
        existing = sb.table("memories").select("id, embedding_id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        rows = _resp_data(existing)
        if rows:
            memory_id = rows[0]["id"]
            existing_embedding = rows[0].get("embedding_id")
            is_duplicate = True
        else:
            is_duplicate = False
            # Decide which DB column to write the text into:
            # default = 'text'; set MEMORIES_TEXT_COLUMN=value if your schema uses 'value' instead
            text_col = (os.getenv("MEMORIES_TEXT_COLUMN") or "text").strip().lower()

            base_payload = {
                "type": mem_type,
                "title": title or None,
                text_col: text,              # write into 'text' or 'value'
                "tags": tags,
                "source": source,
                "role_view": role_view,
                "dedupe_hash": dedupe_hash,
            }

            # INSERT (no .select() chaining)
            sb.table("memories").insert(base_payload).execute()

            # SELECT the new row id
            fetched = sb.table("memories").select("id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
            frows = _resp_data(fetched)
            if not frows:
                raise RuntimeError("Insert succeeded but follow-up select returned no rows")
            memory_id = frows[0]["id"]
            existing_embedding = None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase upsert error: {e}")

    # 2) Embedding (best-effort)
    embedded = False
    embedding_vec = None
    vector_id = None
    try:
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        # Only embed if new OR duplicate without embedding_id set
        if (not is_duplicate) or (is_duplicate and not existing_embedding):
            from openai import OpenAI
            oai = OpenAI()
            kwargs = {"model": embed_model, "input": text}
            if os.getenv("EMBED_DIM"):
                kwargs["dimensions"] = int(os.getenv("EMBED_DIM"))
            eresp = oai.embeddings.create(**kwargs)
            embedding_vec = eresp.data[0].embedding
            embedded = True
    except Exception:
        embedded = False  # DB row still saved

    # 3) Pinecone upsert (best-effort)
    pinecone_upserted = False
    try:
        if embedding_vec:
            from vendors.pinecone_client import get_index
            index = get_index()
            vector_id = f"mem_{memory_id}"
            namespace = {"semantic":"semantic","episodic":"episodic","procedural":"procedural"}[mem_type]
            metadata = {
                "type": mem_type,
                "title": title,
                "tags": tags,
                "created_at": _now_iso(),
                "role_view": role_view,
                "entity_ids": [],
                "source": source,
                "reason": "primary",
            }
            index.upsert(
                vectors=[{"id": vector_id, "values": embedding_vec, "metadata": metadata}],
                namespace=namespace,
            )
            pinecone_upserted = True
            try:
                sb.table("memories").update({"embedding_id": vector_id}).eq("id", memory_id).execute()
            except Exception:
                pass
    except Exception:
        pinecone_upserted = False

    return {
        "memory_id": memory_id,
        "embedding_id": vector_id or existing_embedding,
        "duplicate": is_duplicate,
        "embedded": embedded,
        "pinecone_upserted": pinecone_upserted,
    }

# Canonical PRD path
@router.post("/memories/upsert")
async def memories_upsert(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)

# Aliases for connector-generated names (avoid 404s)
@router.post("/memories_upsert_post", include_in_schema=False)
async def memories_upsert_alias_post(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)

@router.post("/memories_upsert", include_in_schema=False)
async def memories_upsert_alias(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)

@router.post("/memories/upsert/", include_in_schema=False)
async def memories_upsert_slash(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)
