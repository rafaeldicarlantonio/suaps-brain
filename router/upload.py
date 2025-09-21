# router/upload.py
import os, hashlib
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Header, HTTPException, UploadFile, File, Form

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.converters import sniff_and_convert
from ingest.pipeline import normalize_text, chunk_text, upsert_memories_from_chunks

from extractors.signals import extract_signals_from_text
from memory.autosave import apply_autosave

router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),             # csv or leave empty
    type: Optional[str] = Form("semantic"),       # semantic default
    x_api_key: Optional[str] = Header(None),
):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if type not in ("semantic","episodic","procedural"):
        raise HTTPException(status_code=400, detail="type must be one of semantic|episodic|procedural")

    # 1) Read & convert
    raw = await file.read()
    text, mime = sniff_and_convert(file.filename, raw)
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="no text extracted from file")
    text = normalize_text(text)

    sb = get_client()

    # 2) Persist a file row for provenance (non-fatal if table missing)
    try:
        frow = {"filename": file.filename, "mime_type": mime, "bytes": len(raw), "storage_url": "", "text_extracted": text}
        sb.table("files").insert(frow).execute()
        sel = sb.table("files").select("id").order("created_at", desc=True).limit(1).execute()
        fid = (sel.data or [{}])[0].get("id")
    except Exception:
        fid = None

    # 3) Chunk & ingest (original behavior)
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
        text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","value"),
    )

    # 4) Also store fulltext as semantic (safety net)
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
                text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","value"),
            )
        except Exception as e:
            print("Fulltext semantic upsert skipped:", e)

    # 5) Extraction → autosave promotion (quality path)
    extraction_enabled = os.getenv("ENABLE_UPLOAD_SIGNAL_EXTRACTION","true").lower() == "true"
    autosave_summary: Dict[str, Any] = {"saved": False, "items": [], "skipped": []}
    extracted_candidates_count = 0

    if extraction_enabled:
        try:
            ex = extract_signals_from_text(title=file.filename, text=text)
            candidates = ex.get("candidates", []) or []
            extracted_candidates_count = len(candidates)

            # (optional) log to 'signals'
            try:
                for c in candidates:
                    sig_hash = hashlib.md5(
                        (c.get("fact_type","") + (c.get("title") or "") + (c.get("text") or ""))
                        .strip().lower().encode("utf-8")
                    ).hexdigest()
                    sb.table("signals").upsert(
                        {
                            "source": "upload",
                            "source_ref": file.filename,
                            "doc_title": file.filename,
                            "fact_type": c.get("fact_type"),
                            "title": c.get("title"),
                            "text": c.get("text"),
                            "confidence": c.get("confidence"),
                            "hash": sig_hash
                        },
                        on_conflict="hash"
                    ).execute()
            except Exception as le:
                print("signals upsert skipped:", le)

            # promote with your autosave (importance + review flags + save memories)
            autosave_summary = apply_autosave(
                sb=sb,
                pinecone_index=pinecone_index,
                candidates=candidates,
                session_id=None,
                text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","value"),
                author_user_id=None,
            )

            # mark promoted in signals (best-effort)
            try:
                for it in autosave_summary.get("items", []) or []:
                    title_saved = (it.get("title") or "").strip()
                    if title_saved:
                        sb.table("signals").update({"status":"promoted"}).match(
                            {"doc_title": file.filename, "title": title_saved}
                        ).execute()
            except Exception:
                pass

        except Exception as e:
            print("Extractor/autosave failed:", e)

    return {
        "file_id": fid,
        "bytes": len(raw),
        "mime_type": mime,
        "chunks": len(chunks),
        "ingest_result": ingest_result,
        "extraction_enabled": extraction_enabled,
        "extracted_candidates": extracted_candidates_count,
        "autosave": autosave_summary,
    }
