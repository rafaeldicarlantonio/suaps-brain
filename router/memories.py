# router/memories.py
from typing import Optional, List, Dict, Any
import os, re, hashlib, datetime
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()

_ALLOWED_TYPES = {"episodic", "semantic", "procedural"}

def _norm_text(s: str) -> str:
    s = s or ""
    # normalize whitespace and strip page headers-like numbers
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s).strip()
    return s

def _to_tags(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [t.strip() for t in v.split(",") if t.strip()]
    return []

def _to_roles(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [r.strip() for r in v.split(",") if r.strip()]
    return []

def _namespace_for(mem_type: str) -> str:
    # PRD namespaces map 1:1 with type
    return {"semantic": "semantic", "episodic": "episodic", "procedural": "procedural"}.get(mem_type, "semantic")

def _source_of(v: Any) -> str:
    s = (v or "").strip().lower()
    return s if s in {"upload","ingest","chat","wiki"} else "ingest"

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

@router.post("/memories/upsert")
async def memories_upsert_endpoint(
    request: Request,
    x_api_key: Optional[str] = Header(None),
):
    """
    Minimal working upsert:
    1) Validate + normalize payload
    2) Insert memory row (or return existing via dedupe_hash)
    3) Create embedding (best-effort)
    4) Upsert in Pinecone with metadata (best-effort)
    5) Update memories.embedding_id (if vector inserted)
    Returns: {"memory_id", "embedding_id", "duplicate", "pinecone_upserted", "embedded"}
    """
    # --- optional simple auth ---
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Body must be a JSON object")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    mem_type = str(payload.get("type", "")).strip().lower()
    title = (payload.get("title") or "").strip()
    text = _norm_text(payload.get("text") or "")
    tags = _to_tags(payload.get("tags"))
    role_view = _to_roles(payload.get("role_view"))
    source = _source_of(payload.get("source"))

    if mem_type not in _ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"type must be one of {sorted(_ALLOWED_TYPES)}")
    if not text:
        raise HTTPException(status_code=400, detail="text is required and cannot be empty")

    # dedupe hash (normalized text)
    dedupe_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # --- Supabase insert or fetch existing ---
    try:
        from vendors.supabase_client import get_client
        sb = get_client()
    except Exception as e:
        # Without Supabase the endpoint can't fulfill "memory" semantics
        raise HTTPException(status_code=500, detail=f"Supabase client error: {e}")

    # Check duplicate by dedupe_hash
    try:
        existing = sb.table("memories").select("id, embedding_id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        if hasattr(existing, "data"):
            rows = existing.data
        else:
            rows = existing.get("data")  # client variations
        if rows:
            memory_id = rows[0]["id"]
            existing_embedding = rows[0].get("embedding_id")
            # We'll still try to embed/upsert if embedding_id is missing
            is_duplicate = True
        else:
            is_duplicate = False
            # Insert new row
            insert_payload = {
                "type": mem_type,
                "title": title or None,
                "text": text,
                "tags": tags,
                "source": source,
                "role_view": role_view,
                "dedupe_hash": dedupe_hash,
                # created_at defaults in DB; we add updated_at via trigger in later phases
            }
            ins = sb.table("memories").insert(insert_payload).select("id").execute()
            if hasattr(ins, "data"):
                memory_id = ins.data[0]["id"]
            else:
                memory_id = ins["data"][0]["id"]
            existing_embedding = None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase upsert error: {e}")

    # --- Embedding (best-effort) ---
    embedded = False
    embedding_vec = None
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    try:
        # We embed even for duplicates if embedding_id was not set earlier
        if (not is_duplicate) or (is_duplicate and not existing_embedding):
            from openai import OpenAI
            oai = OpenAI()
            eresp = oai.embeddings.create(model=embed_model, input=text)
            embedding_vec = eresp.data[0].embedding  # list[float]
            embedded = True
    except Exception as e:
        # Keep going; we'll return embedded=False
        embedded = False

    # --- Pinecone upsert (best-effort) ---
    pinecone_upserted = False
    vector_id = None
    try:
        if embedding_vec:
            namespace = _namespace_for(mem_type)
            vector_id = f"mem_{memory_id}"
            # Get index via your helper (preferred)
            try:
                from vendors.pinecone_client import get_index
                index = get_index()
            except Exception:
                # Fallback direct client if helper not available
                from pinecone import Pinecone
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                index = pc.Index(os.getenv("PINECONE_INDEX"))

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
            # Upsert
            index.upsert(
                vectors=[{"id": vector_id, "values": embedding_vec, "metadata": metadata}],
                namespace=namespace,
            )
            pinecone_upserted = True

            # Write back embedding_id
            try:
                sb.table("memories").update({"embedding_id": vector_id}).eq("id", memory_id).execute()
            except Exception:
                pass
    except Exception:
        pinecone_upserted = False

    # --- Response ---
    return JSONResponse(
        {
            "memory_id": memory_id,
            "embedding_id": vector_id,
            "duplicate": is_duplicate,
            "embedded": embedded,
            "pinecone_upserted": pinecone_upserted,
        },
        status_code=200,
    )
