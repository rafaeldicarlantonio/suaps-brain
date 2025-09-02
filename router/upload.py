# router/upload.py
import os, io, time, mimetypes
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, Header, HTTPException
from agent import store, retrieval
from agent.ingest import clean_text, distill_chunk, ingest_text
from vendors.supabase_client import supabase

router = APIRouter()

def auth(x_api_key: Optional[str]):
    want = os.getenv("ACTIONS_API_KEY") or "dev_key"
    if (os.getenv("DISABLE_AUTH","false").lower() == "true"):
        return
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="invalid X-API-Key")

def _read_file(file: UploadFile) -> str:
    name = file.filename or ""
    mime = file.content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    raw = file.file.read()
    # naive handling for MVP
    if mime in ("text/markdown","text/plain","application/json"):
        return raw.decode("utf-8", errors="ignore")
    if mime in ("application/pdf",):
        try:
            import pypdf
            from io import BytesIO
            r = pypdf.PdfReader(BytesIO(raw))
            return "\n".join([p.extract_text() or "" for p in r.pages])
        except Exception:
            return raw.decode("utf-8","ignore")
    if mime in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"):
        try:
            import docx
            d = docx.Document(io.BytesIO(raw))
            return "\n".join([p.text for p in d.paragraphs])
        except Exception:
            return raw.decode("utf-8","ignore")
    # fallback
    return raw.decode("utf-8","ignore")

@router.post("/upload")
async def upload(x_api_key: Optional[str] = Header(None), file: UploadFile = File(...), tags: Optional[str] = Form(None)):
    auth(x_api_key)
    t0 = time.time()
    txt = _read_file(file)
    txt = clean_text(txt)
    # persist file row (MVP)
    try:
        meta = {"filename": file.filename, "mime_type": file.content_type, "bytes": len(txt.encode("utf-8")), "text_extracted": txt[:1000000]}
        fr = supabase.table("files").insert(meta).execute()
        file_id = fr.data[0]["id"] if fr.data else None
    except Exception:
        file_id = None

    # Distill and store memory (semantic)
    base_tags = [t.strip() for t in (tags or "").split(",") if t.strip()] if tags else []
    ids = distill_chunk(user_id=None, raw_text=txt, base_tags=base_tags, make_qa=False)
    return {"file_id": file_id, "bytes": len(txt.encode("utf-8")), "mime_type": file.content_type, "ingest_ids": ids, "latency_ms": int((time.time()-t0)*1000)}

# --- OCR helpers (optional) ---
import os, io, logging
logger = logging.getLogger(__name__)
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None
try:
    from PIL import Image
except Exception:
    Image = None

def text_density(text: str, raw: bytes) -> float:
    try:
        return (len((text or "").strip()) / max(1, len(raw)))
    except Exception:
        return 1.0

def try_pypdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(raw))
        pages = [p.extract_text() or "" for p in r.pages]
        return "\n\n".join(pages).strip()
    except Exception as ex:
        logger.warning("pypdf extract failed: %s", ex)
        return ""

def ocr_pdf_bytes_with_tesseract(raw: bytes) -> str:
    """Requires pdf2image + Pillow + pytesseract; system deps: poppler, tesseract-ocr.
       Set TESSDATA_PREFIX on Render if needed."""
    if pytesseract is None or convert_from_bytes is None:
        logger.warning("OCR requested but pytesseract/pdf2image not available")
        return ""
    try:
        images = convert_from_bytes(raw, dpi=300)  # needs poppler on the box
        out = []
        for im in images:
            txt = pytesseract.image_to_string(im)
            if txt:
                out.append(txt)
        return "\n\n".join(out).strip()
    except Exception as ex:
        logger.warning("OCR pipeline failed: %s", ex)
        return ""

