# ingest/converters.py
import io, os, re
from typing import Tuple
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

def _norm_text(s: str) -> str:
    s = s or ""
    # remove common cruft
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    # drop repeating page footers/headers (very light heuristic)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def from_pdf(data: bytes) -> Tuple[str, str]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    txt = "\n\n".join(pages)
    return _norm_text(txt), "application/pdf"

def from_docx(data: bytes) -> Tuple[str, str]:
    f = io.BytesIO(data)
    doc = Document(f)
    txt = "\n".join([p.text for p in doc.paragraphs])
    return _norm_text(txt), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

def from_md(data: bytes) -> Tuple[str, str]:
    txt = data.decode("utf-8", errors="ignore")
    # strip yaml frontmatter
    txt = re.sub(r"^---\n.*?\n---\n", "", txt, flags=re.DOTALL)
    return _norm_text(txt), "text/markdown"

def from_html(data: bytes) -> Tuple[str, str]:
    html = data.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    txt = soup.get_text("\n")
    return _norm_text(txt), "text/html"

def sniff_and_convert(filename: str, data: bytes) -> Tuple[str, str]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return from_pdf(data)
    if name.endswith(".docx"):
        return from_docx(data)
    if name.endswith(".md") or name.endswith(".markdown"):
        return from_md(data)
    if name.endswith(".html") or name.endswith(".htm"):
        return from_html(data)
    # fallback: assume utf-8 text
    return _norm_text(data.decode("utf-8", errors="ignore")), "text/plain"
