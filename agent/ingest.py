"""
agent/ingest.py  — self-contained ingestion that ALWAYS persists.

Pipeline (PRD §6):
- clean_text  -> normalize
- chunk_text  -> ~1000 token chunks
- distill_chunk -> title/summary/tags (tolerant)
- dedupe -> exact sha256 + 64-bit SimHash near-dup
- INSERT row in Supabase -> memories
- EMBED with OpenAI -> Pinecone upsert (namespace = type)
- UPDATE memories.embedding_id = "mem_<id>"
"""

from __future__ import annotations

import os, re, json, hashlib, logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------- Vendors (required) ----------
from vendors.openai_client import client as _oai_client, EMBED_MODEL, CHAT_MODEL  # env must have OPENAI_API_KEY
from vendors.supabase_client import supabase                                       # must be configured
from vendors.pinecone_client import get_index

# ---------- Optional project modules (if present) ----------
try:
    from agent import entities as _entities
except Exception:
    _entities = None

# ---------- Config ----------
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "uap-kb")
TARGET_TOKENS = 1000
OVERLAP_TOKENS = 120
NEAR_DUP_SIMHASH_BITS = 64
NEAR_DUP_THRESHOLD = 0.92
MAX_TAGS = 8


# ============ Utilities ============
def _approx_tokens(s: str) -> int:
    return max(1, int(len(s or "") / 4))

_HEADER_FOOTER_RE = re.compile(r"^(page \d+|\d+)$", re.IGNORECASE)
_MULTISPACE_RE    = re.compile(r"[ \t\u00A0\x0b\x0c]+")
_MULTINEWLINE_RE  = re.compile(r"\n{3,}")

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            lines.append("")
            continue
        if _HEADER_FOOTER_RE.match(s.lower()):
            continue
        lines.append(ln)
    t = "\n".join(lines)
    t = _MULTISPACE_RE.sub(" ", t)
    t = _MULTINEWLINE_RE.sub("\n\n", t)
    t = t.replace("\ufeff", "").replace("\u200b", "")
    return t.strip()

_HEADING_RE   = re.compile(r"^(#{1,6}\s+.+|\s*[A-Z][A-Z0-9 \-]{6,}\s*)$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

def _split_headings(text: str) -> List[Tuple[str, str]]:
    lines = text.split("\n")
    chunks: List[Tuple[str, List[str]]] = []
    cur_head = ""
    cur_buf: List[str] = []
    for ln in lines:
        if _HEADING_RE.match(ln.strip()):
            if cur_buf:
                chunks.append((cur_head, cur_buf)); cur_buf = []
            cur_head = ln.strip("# ").strip()
        else:
            cur_buf.append(ln)
    if cur_buf:
        chunks.append((cur_head, cur_buf))
    return [(h, "\n".join(buf).strip()) for h, buf in chunks]

def _split_paragraphs(section_text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", section_text) if p.strip()]

def _split_sentences(paragraph: str) -> List[str]:
    return _SENT_SPLIT_RE.split(paragraph.strip()) if paragraph.strip() else []

def chunk_text(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[Dict[str, Any]]:
    sections = _split_headings(text)
    chunks: List[Dict[str, Any]] = []

    for heading, sec in sections:
        paragraphs = _split_paragraphs(sec)
        buf: List[str] = []; buf_tokens = 0; start_tok = 0

        def _flush():
            nonlocal buf, buf_tokens, start_tok
            if not buf:
                return
            txt = "\n".join(buf).strip()
            if not txt:
                buf, buf_tokens = [], 0
                return
            end_tok = start_tok + _approx_tokens(txt)
            chunks.append({"text": txt, "heading": heading, "start_token": start_tok,
                           "end_token": end_tok, "approx_tokens": _approx_tokens(txt)})
            if overlap_tokens > 0:
                tail = txt[-overlap_tokens*4:]; buf = [tail]
                buf_tokens = _approx_tokens(tail); start_tok = end_tok - buf_tokens
            else:
                buf, buf_tokens = [], 0; start_tok = end_tok

        for para in paragraphs:
            for s in _split_sentences(para):
                tok = _approx_tokens(s)
                if buf_tokens + tok > target_tokens:
                    _flush()
                buf.append(s); buf_tokens += tok
            if buf_tokens >= target_tokens * 0.9:
                _flush()
        _flush()

    # merge tiny trailers
    merged: List[Dict[str, Any]] = []
    for c in chunks:
        if merged and c["approx_tokens"] < min(200, int(0.25 * target_tokens)):
            merged[-1]["text"] = (merged[-1]["text"].rstrip() + "\n\n" + c["text"].lstrip()).strip()
            merged[-1]["approx_tokens"] = _approx_tokens(merged[-1]["text"])
            merged[-1]["end_token"] = merged[-1]["start_token"] + merged[-1]["approx_tokens"]
        else:
            merged.append(c)
    return merged


# ============ Distill (tolerant) ============
def distill_chunk(
    text: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **kwargs,
):
    # recover alt keys
    if text is None:
        text = kwargs.get("chunk_text") or kwargs.get("content") or kwargs.get("body")
    if not text:
        return {"title": title or "Untitled", "summary": "", "tags": (tags or [])[:MAX_TAGS]}

    # heuristic if needed
    if _oai_client is None:
        first = text.strip().split("\n", 1)[0][:120]
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", text.lower())
        freq = {}; [freq.setdefault(w, 0) for _ in words]
        for w in words: freq[w] = freq.get(w, 0) + 1
        ht = [w for w,_ in sorted(freq.items(), key=lambda t: (-t[1], t[0]))[:5]]
        return {"title": title or first, "summary": text[:500], "tags": (ht or (tags or []))[:MAX_TAGS]}

    prompt = (
        "Summarize the following chunk for a knowledge base.\n"
        "Return JSON with keys: title, summary, tags (<=8 short tags).\n"
        "Text:\n---\n" + text[:8000] + "\n---"
    )
    try:
        resp = _oai_client.chat.completions.create(
            model=CHAT_MODEL, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        js = json.loads(resp.choices[0].message.content)
        return {
            "title": js.get("title") or title or text.strip().split("\n", 1)[0][:120],
            "summary": js.get("summary") or text[:500],
            "tags": (js.get("tags") or tags or [])[:MAX_TAGS],
        }
    except Exception:
        first = text.strip().split("\n", 1)[0][:120]
        return {"title": title or first, "summary": text[:500], "tags": (tags or [])[:MAX_TAGS]}


# ============ Dedupe ============
def _normalize_for_hash(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())

def compute_dedupe_hash(s: str) -> str:
    return hashlib.sha256(_normalize_for_hash(s).encode("utf-8")).hexdigest()

def _tokenize_for_simhash(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]{3,}", _normalize_for_hash(s))

def simhash64(s: str) -> int:
    tokens = _tokenize_for_simhash(s)
    if not tokens:
        return 0
    v = [0] * NEAR_DUP_SIMHASH_BITS
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(NEAR_DUP_SIMHASH_BITS):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i, val in enumerate(v):
        if val >= 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def is_near_duplicate(sim_a: int, sim_b: int) -> bool:
    max_dist = max(0, int(NEAR_DUP_SIMHASH_BITS * (1.0 - NEAR_DUP_THRESHOLD)))  # 64*(1-0.92)=5
    return hamming(sim_a, b=sim_b) <= max_dist


# ============ Core ingest ============
def _embed_vec(text: str) -> List[float]:
    r = _oai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding

def _pinecone_index():
    return get_index()


def _upsert_vector(mem_id: str, vec: List[float], namespace: str, meta: Dict[str, Any]) -> None:
    idx = _pinecone_index()
    idx.upsert(
        vectors=[{"id": f"mem_{mem_id}", "values": vec, "metadata": meta}],
        namespace=namespace,
    )

def ingest_text(
    *,
    title: Optional[str],
    text: str,
    type: str,
    tags: Optional[List[str]] = None,
    source: Optional[str] = None,
    role_view: Optional[List[str]] = None,
    file_id: Optional[str] = None,
    session_id: Optional[str] = None,
    dedupe: bool = True,
) -> Dict[str, Any]:
    assert type in ("episodic", "semantic", "procedural"), "invalid type"
    norm = clean_text(text)
    chunks = chunk_text(norm)

    batch_simhashes: List[int] = []
    upserted: List[Dict[str, Any]] = []
    skipped:  List[Dict[str, Any]] = []

    for ch in chunks:
        ctext = ch["text"].strip()
        if not ctext:
            continue

        # exact dedupe (DB-level)
        dh = compute_dedupe_hash(ctext)
        if dedupe:
            try:
                q = supabase.table("memories").select("id").eq("dedupe_hash", dh).limit(1).execute()
                if q.data:
                    skipped.append({"reason": "duplicate", "dedupe_hash": dh})
                    continue
            except Exception as ex:
                logger.warning("dedupe check failed: %s", ex)

        # in-batch near-dup
        sh = simhash64(ctext)
        if dedupe and any(is_near_duplicate(sh, prev) for prev in batch_simhashes):
            skipped.append({"reason": "near-duplicate"})
            continue
        batch_simhashes.append(sh)

        # distill
        meta = distill_chunk(text=ctext, title=title, tags=tags or [])
        ctitle = meta.get("title") or (title or "Untitled")
        ctags  = list(dict.fromkeys((tags or []) + (meta.get("tags") or [])))[:MAX_TAGS]

        # INSERT memory row
        try:
            ins = supabase.table("memories").insert({
                "type": type,
                "title": ctitle,
                "text": ctext,
                "tags": ctags,
                "source": source or "ingest",
                "file_id": file_id,
                "session_id": session_id,
                "role_view": role_view or [],
                "dedupe_hash": dh,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            mem_id = ins.data[0]["id"]
        except Exception as ex:
            logger.error("Supabase insert failed: %s", ex)
            skipped.append({"reason": "store-failed", "error": str(ex)})
            continue

        # (optional) entities
        entity_ids: List[str] = []
        if _entities and hasattr(_entities, "extract_entities"):
            try:
                ents = _entities.extract_entities(ctext) or []
                # If returns list of names/ids tuples/dicts—normalize to strings
                cleaned = []
                for e in ents:
                    if isinstance(e, str):
                        cleaned.append(e)
                    elif isinstance(e, (list, tuple)) and e:
                        cleaned.append(str(e[0]))
                    elif isinstance(e, dict):
                        cleaned.append(e.get("id") or e.get("name"))
                entity_ids = [x for x in cleaned if x]
            except Exception as ex:
                logger.warning("entities failed: %s", ex)

        # EMBED + UPSERT vector
        try:
            vec = _embed_vec(ctext)
            meta_vec = {
                "type": type,
                "title": ctitle,
                "tags": ctags,
                "created_at": datetime.utcnow().isoformat(),
                "role_view": role_view or [],
                "entity_ids": entity_ids,
                "source": source or "ingest",
            }
            _upsert_vector(mem_id, vec, namespace=type, meta=meta_vec)
            # update embedding_id for reverse lookup
            try:
                supabase.table("memories").update({"embedding_id": f"mem_{mem_id}"}).eq("id", mem_id).execute()
            except Exception:
                pass
        except Exception as ex:
            logger.error("Upsert vector failed: %s", ex)
            skipped.append({"reason": "vector-failed", "error": str(ex)})
            continue

        upserted.append({"memory_id": mem_id, "embedding_id": f"mem_{mem_id}"})

    return {"upserted": upserted, "skipped": skipped}


def ingest_batch(items: List[Dict[str, Any]], dedupe: bool = True) -> Dict[str, Any]:
    all_up: List[Dict[str, Any]] = []
    all_sk: List[Dict[str, Any]] = []
    for it in items:
        res = ingest_text(
            title      = it.get("title"),
            text       = it.get("text") or "",
            type       = it.get("type") or "semantic",
            tags       = it.get("tags") or [],
            source     = it.get("source") or "ingest",
            role_view  = it.get("role_view") or [],
            file_id    = it.get("file_id"),
            session_id = it.get("session_id"),
            dedupe     = dedupe if it.get("dedupe", None) is None else bool(it["dedupe"]),
        )
        all_up.extend(res.get("upserted", []))
        all_sk.extend(res.get("skipped", []))
    return {"upserted": all_up, "skipped": all_sk}
