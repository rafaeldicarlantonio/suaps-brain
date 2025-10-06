# router/upload.py
import os, hashlib, json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.converters import sniff_and_convert
from ingest.pipeline import normalize_text, chunk_text, upsert_memories_from_chunks

from extractors.signals import extract_signals_from_text
from memory.autosave import apply_autosave

router = APIRouter()

def _safe_len(x) -> int:
    try:
        return len(x)  # list/dict/str
    except Exception:
        return 0

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),             # csv or leave empty
    type: Optional[str] = Form("semantic"),       # semantic default
    extract_signals: Optional[bool] = Form(True), # <— NEW: control from Make
    x_api_key: Optional[str] = Header(None),
):
    # --- auth guard ---
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if type not in ("semantic","episodic","procedural"):
        raise HTTPException(status_code=400, detail="type must be one of semantic|episodic|procedural")

    # --- read & convert ---
    raw = await file.read()
    try:
        text, mime = sniff_and_convert(file.filename, raw)
    except ValueError as e:
        # make converters raise ValueError for bad bytes (invalid PDF, etc.)
        raise HTTPException(status_code=400, detail=f"Invalid content: {e}")
    except Exception as e:
        # unknown conversion failure
        raise HTTPException(status_code=400, detail=f"Conversion failed: {e}")

    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="no text extracted from file")
    text = normalize_text(text)

    sb = get_client()

    # --- provenance: files row (best-effort) ---
    fid = None
    try:
        frow = {
            "filename": file.filename, "mime_type": mime, "bytes": len(raw),
            "storage_url": "", "text_extracted": text
        }
        sb.table("files").insert(frow).execute()
        sel = sb.table("files").select("id").order("created_at", desc=True).limit(1).execute()
        fid = (sel.data or [{}])[0].get("id")
    except Exception:
        pass

    # --- chunk & ingest ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","2000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP","200"))
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    pinecone_index = get_index()
    tags_list: List[str] = [t.strip() for t in (tags or "").split(",") if t.strip()]

    ingest_result = upsert_memories_from_chunks(
        sb=sb,
        pinecone_index=pinecone_index,
        embedder=None,
        file_id=fid,
        title_prefix=file.filename,
        chunks=chunks,
        mem_type=type,
        tags=tags_list,
        role_view=[],
        source="upload",
        text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","text"),
    )

    # Optional: also store fulltext as semantic
    if os.getenv("UPLOAD_ALSO_STORE_FULLTEXT_SEMANTIC","true").lower() == "true" and type != "semantic":
        try:
            upsert_memories_from_chunks(
                sb=sb,
                pinecone_index=pinecone_index,
                embedder=None,
                file_id=fid,
                title_prefix=file.filename,
                chunks=[text],
                mem_type="semantic",
                tags=list(set((tags_list or []) + ["fulltext"])),
                role_view=[],
                source="upload",
                text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","text"),
            )
        except Exception as e:
            print("Fulltext semantic upsert skipped:", e)

    # --- extraction → autosave (tolerant) ---
    autosave_summary: Dict[str, Any] = {"saved": False, "items": [], "skipped": []}
    autosave_error: Optional[str] = None
    extracted_candidates_count = 0

    if (os.getenv("ENABLE_UPLOAD_SIGNAL_EXTRACTION","true").lower() == "true") and extract_signals:
        try:
            ex = extract_signals_from_text(title=file.filename, text=text) or {}
            # tolerate various shapes: {"candidates":[...]} or {"items":[...]}
            candidates = ex.get("candidates") or ex.get("items") or []
            if not isinstance(candidates, list):
                # sometimes models return a stringified JSON – try to parse
                if isinstance(candidates, str):
                    try:
                        candidates = json.loads(candidates)
                    except Exception:
                        candidates = []
                else:
                    candidates = []
            extracted_candidates_count = len(candidates)

            # best-effort signals logging
            try:
                for c in candidates:
                    sig_hash = hashlib.md5(
                        (str(c.get("fact_type","")) + (c.get("title") or "") + (c.get("text") or ""))
                        .strip().lower().encode("utf-8")
                    ).hexdigest()
                    sb.table("signals").upsert(
                        {
                            "source": "upload",
                            "source_ref": file.filename,
                            "doc_title": file.filename,
                            "fact_type": c.get("fact_type") or c.get("type"),
                            "title": c.get("title"),
                            "text": c.get("text") or c.get("value"),
                            "confidence": c.get("confidence"),
                            "hash": sig_hash
                        },
                        on_conflict="hash"
                    ).execute()
            except Exception as le:
                print("signals upsert skipped:", le)

            try:
                autosave_summary = apply_autosave(
                    sb=sb,
                    pinecone_index=pinecone_index,
                    candidates=candidates,
                    session_id=None,
                    text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","text"),
                    author_user_id=None,
                ) or autosave_summary
            except Exception as ae:
                # DO NOT fail the request – surface as warning
                autosave_error = str(ae)
                print("Autosave failed:", autosave_error)

        except Exception as e:
            autosave_error = str(e)
            print("Extractor/autosave failed:", autosave_error)

        # --- stable response for Make/Slack/Email ---
    # Try to extract common counts from ingest_result without assuming shape
    def _get(d: Dict[str, Any], key: str) -> int:
        try:
            v = d.get(key, [])
            return len(v) if isinstance(v, list) else int(v or 0)
        except Exception:
            return 0

    upserted = _get(ingest_result, "upserted")
    updated  = _get(ingest_result, "updated")
    skipped  = _get(ingest_result, "skipped")

    return {
        "status": "ok",
        "file_name": file.filename,
        "file_id": fid,
        "mime_type": mime,
        "bytes": len(raw),
        "chunks": len(chunks),
        "ingest": {
            "upserted": upserted,
            "updated": updated,
            "skipped": skipped,
            "raw": ingest_result,  # keep original for debugging
        },
        "extraction_enabled": bool(extract_signals),
        "extracted_candidates": extracted_candidates_count,
        "autosave": autosave_summary,
        "autosave_error": autosave_error,   # null if all good
    }
