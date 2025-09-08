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
    return s if s in {"upload","ingest","chat","wiki"} else "chat"

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def _resp_data(resp) -> List[Dict[str, Any]]:
    """Normalize supabase response .data across versions."""
    if resp is None: return []
    if hasattr(resp, "data"):  # supabase-py v2
        return resp.data or []
    if isinstance(resp, dict):  # ultra-conservative fallback
        return resp.get("data") or []
    return []

async def _memories_upsert_core(request: Request, x_api_key: Optional[str]) -> Dict[str, Any]:
    # ---- optional API key ----
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ---- read JSON ----
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Body must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    mem_type = str(payload.get("type", "")).strip().lower()
    title = (payload.get("title") or "").strip()
    # Accept either 'text' or 'value' in the request body, prefer non-empty.
    raw_text = payload.get("value") or payload.get("text") or ""
    text = _norm_text(raw_text)
    tags = _to_list(payload.get("tags"))
    role_view = _to_list(payload.get("role_view"))
    _role_metadata = payload.get("role_metadata")  # tolerated, ignored
    source = _source_of(payload.get("source"))

    if mem_type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"type must be one of {sorted(_ALLOWED_TYPES)}")
    if not text:
        raise HTTPException(status_code=400, detail="Either 'text' or 'value' must be provided and non-empty")

    dedupe_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ---- Supabase client ----
    try:
        from vendors.supabase_client import get_client
        sb = get_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase client not available: {e}")

    # ---- duplicate check ----
    try:
        existing = sb.table("memories").select("id, embedding_id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        rows = _resp_data(existing)
        if rows:
            memory_id = rows[0]["id"]
            existing_embedding = rows[0].get("embedding_id")
            is_duplicate = True
        else:
            is_duplicate = False
            # We don't know whether your DB column is 'text' or 'value'. Try smartly.
            # 1) Pick default from env, else try 'text' first.
            preferred_col = os.getenv("MEMORIES_TEXT_COLUMN", "text").strip().lower()
            candidate_cols = [preferred_col] + (["value"] if preferred_col != "value" else []) + (["text"] if preferred_col != "text" else [])
            # Build static parts
            base_payload = {
                "type": mem_type,
                "title": title or None,
                "tags": tags,
                "source": source,
                "role_view": role_view,
                "dedupe_hash": dedupe_hash,
            }
            insert_ok = False
            last_err = None

            for col in candidate_cols:
                try:
                    ins_payload = dict(base_payload)
                    ins_payload[col] = text  # try with this column name
                    sb.table("memories").insert(ins_payload).execute()
                    insert_ok = True
                    used_col = col
                    break
                except Exception as e:
                    # Common failures we expect: "null value in column 'value'..." when we used the wrong col,
                    # or "column 'text' does not exist" when schema differs. Try the other column next.
                    last_err = e

            if not insert_ok:
                raise HTTPException(status_code=500, detail=f"Supabase insert failed (tried columns {candidate_cols}): {last_err}")

            # Fetch the inserted row id
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
            if os.getenv("EMBED_DIM"):
                kwargs["dimensions"] = int(os.getenv("EMBED_DIM"))
            eresp = oai.embeddings.create(**kwargs)
            embedding_vec = eresp.data[0].embedding
            embedded = True
    except Exception:
        embedded = False  # keep going

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
