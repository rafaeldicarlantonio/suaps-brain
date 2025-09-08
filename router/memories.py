from typing import Optional, List, Any, Dict
import os, re, hashlib, datetime
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()
_ALLOWED_TYPES = {"episodic", "semantic", "procedural"}

def _norm_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s).strip()
    return s

def _to_list(v: Any) -> List[str]:
    if v is None: return []
    if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):  return [t.strip() for t in v.split(",") if t.strip()]
    return []

def _source_of(v: Any) -> str:
    s = (v or "").strip().lower()
    return s if s in {"upload","ingest","chat","wiki"} else "ingest"

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

# --- core logic shared by all route aliases ---
async def _memories_upsert_core(request: Request, x_api_key: Optional[str]) -> Dict[str, Any]:
    # ---- optional simple auth ----
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---- read JSON, tolerate extra fields like 'role_metadata' ----
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Body must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    mem_type = str(payload.get("type", "")).strip().lower()
    title = (payload.get("title") or "").strip()
    text = _norm_text(payload.get("text") or "")
    tags = _to_list(payload.get("tags"))
    role_view = _to_list(payload.get("role_view"))
    # Accept but ignore role_metadata if the connector sends it:
    _role_metadata = payload.get("role_metadata")  # ignored for MVP
    source = _source_of(payload.get("source"))

    if mem_type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"type must be one of {sorted(_ALLOWED_TYPES)}")
    if not text:
        raise HTTPException(status_code=400, detail="text is required and cannot be empty")

    dedupe_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ---- Supabase insert or fetch existing ----
    try:
        from vendors.supabase_client import get_client
        sb = get_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase client not available: {e}")

    try:
        existing = sb.table("memories").select("id, embedding_id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        rows = existing.data if hasattr(existing, "data") else existing.get("data")
        if rows:
            memory_id = rows[0]["id"]
            existing_embedding = rows[0].get("embedding_id")
            is_duplicate = True
        else:
            is_duplicate = False
            insert_payload = {
                "type": mem_type,
                "title": title or None,
                "text": text,
                "tags": tags,
                "source": source,
                "role_view": role_view,
                "dedupe_hash": dedupe_hash,
            }
            ins = sb.table("memories").insert(insert_payload).select("id").execute()
            memory_id = ins.data[0]["id"] if hasattr(ins, "data") else ins["data"][0]["id"]
            existing_embedding = None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase upsert error: {e}")

    # ---- Embedding (best-effort) ----
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
            # Optional: allow EMBED_DIM to keep index dimension consistent
            if os.getenv("EMBED_DIM"):
                kwargs["dimensions"] = int(os.getenv("EMBED_DIM"))
            eresp = oai.embeddings.create(**kwargs)
            embedding_vec = eresp.data[0].embedding
            embedded = True
    except Exception:
        embedded = False  # keep going; we still saved the row

    # ---- Pinecone upsert (best-effort) ----
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

# --- canonical PRD path ---
@router.post("/memories/upsert")
async def memories_upsert(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)

# --- aliases for connector-generated names (avoid 404s) ---
@router.post("/memories_upsert_post", include_in_schema=False)
async def memories_upsert_alias_post(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)

@router.post("/memories_upsert", include_in_schema=False)
async def memories_upsert_alias(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)

@router.post("/memories/upsert/", include_in_schema=False)
async def memories_upsert_slash(request: Request, x_api_key: Optional[str] = Header(None)):
    return JSONResponse(await _memories_upsert_core(request, x_api_key), status_code=200)
