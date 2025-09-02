# router/upload.py
import os, io, time, mimetypes
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, Header, HTTPException

from vendors.supabase_client import supabase
from agent.ingest import clean_text, ingest_text
from agent import store

router = APIRouter()

# ---------------------------------------------------------------------
# Auth (PRD §15)
# ---------------------------------------------------------------------
def auth(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY") or "dev_key"
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="invalid X-API-Key")

# ---------------------------------------------------------------------
# PDF helpers (pypdf, optional OCR)
# ---------------------------------------------------------------------
def try_pypdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    except Exception:
        return ""

def text_density(txt: str, raw: bytes) -> float:
    if not raw:
        return 0.0
    return len(txt.strip()) / max(1.0, float(len(raw)))

def ocr_pdf_bytes_with_tesseract(raw: bytes) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(raw)
        out = []
        for im in pages:
            out.append(pytesseract.image_to_string(im))
        return "\n".join(out)
    except Exception:
        return ""

# ---------------------------------------------------------------------
# File normalization (PRD §6.1)
# ---------------------------------------------------------------------
def normalize_bytes_to_text(raw: bytes, mime: str) -> str:
    if mime in ("text/markdown","text/plain","application/json"):
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    if mime == "application/pdf":
        # 1) Text layer via pypdf
        text = try_pypdf(raw)
        # 2) Optional OCR if density low and explicitly enabled
        if text_density(text, raw) < float(os.getenv("OCR_DENSITY_THRESHOLD","0.05")) and \
           os.getenv("ENABLE_PDF_OCR","false").lower() == "true":
            text = ocr_pdf_bytes_with_tesseract(raw) or text
        return text
    if mime in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    # Fallback: best-effort UTF-8
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------
@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None),
):
    auth(x_api_key)
    raw = await file.read()
    mime = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"

    # Normalize to text
    extracted = normalize_bytes_to_text(raw, mime)
    norm = clean_text(extracted)

    # Persist file row
    frow = {
        "filename": file.filename,
        "mime_type": mime,
        "bytes": len(raw),
        "storage_url": "",
        "text_extracted": norm,
    }
    ins = supabase.table("files").insert(frow).execute()
    file_id = ins.data[0]["id"]

    # Ingest
    tag_list: List[str] = [t.strip() for t in (tags or "").split(",") if t.strip()]
    ingest_res = ingest_text(
        text=norm,
        type="semantic",
        title=file.filename,
        tags=tag_list,
        source="upload",
        role_view=None,
        file_id=file_id,
        dedupe=True,
    )

    return {
        "file_id": file_id,
        "bytes": len(raw),
        "mime_type": mime,
        "ingest_job_id": ingest_res.get("job_id") if isinstance(ingest_res, dict) else None,
    }
