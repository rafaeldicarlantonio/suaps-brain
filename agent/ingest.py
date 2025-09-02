
"""
agent/ingest.py
----------------
PRD §6 ingestion improvements:
- clean_text(): normalize to UTF-8, collapse whitespace, strip headers/footers/page numbers
- chunk_text(): target ~800–1200 tokens, 100–150 overlap; prefer headings then paragraphs then sentences
- distill_chunk(): LLM pass to produce {title, summary, tags[]} (+ candidate entities when available)
- dedupe: exact (sha256 on normalized text) and near-dup via 64-bit SimHash (threshold >= 0.92)
- store + embed + upsert to Pinecone (via agent.retrieval.upsert_memory_vector)
- hooks for entities & graph (best-effort if agent.entities exists)
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------
try:
    from vendors.openai_client import client as _oai_client
    CHAT_MODEL = os.getenv("EXTRACTOR_MODEL", os.getenv("CHAT_MODEL", "gpt-4.1-mini"))
except Exception:
    _oai_client = None
    CHAT_MODEL = "gpt-4.1-mini"

try:
    from agent import store
except Exception:
    store = None

try:
    from agent import retrieval
except Exception:
    retrieval = None

try:
    from agent import entities as _entities
except Exception:
    _entities = None

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TARGET_TOKENS = 1000  # aim 800–1200
OVERLAP_TOKENS = 120  # 100–150
NEAR_DUP_SIMHASH_BITS = 64
NEAR_DUP_THRESHOLD = 0.92  # PRD: ≥ 0.92 -> considered duplicate
MAX_TAGS = 8

def _approx_tokens(s: str) -> int:
    return max(1, int(len(s or "")/4))

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
_HEADER_FOOTER_RE = re.compile(r'^(page \\d+|\\d+)$', re.IGNORECASE)
_MULTISPACE_RE = re.compile(r'[ \\t\\u00A0\\x0b\\x0c]+')
_MULTINEWLINE_RE = re.compile(r'\\n{3,}')

def clean_text(text: str) -> str:
    if not text:
        return ""
    # unify newlines
    t = text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
    # drop lines that look like headers/footers/page numbers
    lines = []
    for ln in t.split('\\n'):
        s = ln.strip()
        if not s:
            lines.append('')
            continue
        if _HEADER_FOOTER_RE.match(s.lower()):
            continue
        lines.append(ln)
    t = '\\n'.join(lines)
    # collapse spaces
    t = _MULTISPACE_RE.sub(' ', t)
    # collapse multiple blank lines
    t = _MULTINEWLINE_RE.sub('\\n\\n', t)
    # strip BOM / zero-width
    t = t.replace('\\ufeff', '').replace('\\u200b', '')
    return t.strip()

# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------
_HEADING_RE = re.compile(r'^(#{1,6}\\s+.+|\\s*[A-Z][A-Z0-9 \\-]{6,}\\s*)$')  # markdown or SCREAMING headings
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\\s+(?=[A-Z0-9(])')

def _split_headings(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (heading, section_text). Heading may be '' for preface.
    """
    lines = text.split('\\n')
    chunks: List[Tuple[str, List[str]]] = []
    cur_head = ''
    cur_buf: List[str] = []
    for ln in lines:
        if _HEADING_RE.match(ln.strip()):
            # flush previous
            if cur_buf:
                chunks.append((cur_head, cur_buf))
                cur_buf = []
            cur_head = ln.strip('# ').strip()
        else:
            cur_buf.append(ln)
    if cur_buf:
        chunks.append((cur_head, cur_buf))
    return [(h, '\\n'.join(buf).strip()) for h, buf in chunks]

def _split_paragraphs(section_text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r'\\n\\s*\\n+', section_text) if p.strip()]
    return parts

def _split_sentences(paragraph: str) -> List[str]:
    return _SENT_SPLIT_RE.split(paragraph.strip()) if paragraph.strip() else []

def chunk_text(text: str, target_tokens: int = TARGET_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[Dict[str, Any]]:
    """
    Returns list of chunk dicts: {text, heading, start_token, end_token, approx_tokens}
    Prefers boundaries: heading > paragraphs > sentences with overlap.
    """
    sections = _split_headings(text)
    chunks: List[Dict[str, Any]] = []

    for heading, sec in sections:
        paragraphs = _split_paragraphs(sec)
        buf: List[str] = []
        buf_tokens = 0
        start_tok = 0

        def _flush():
            nonlocal buf, buf_tokens, start_tok
            if not buf:
                return
            txt = '\\n'.join(buf).strip()
            if not txt:
                buf, buf_tokens = [], 0
                return
            end_tok = start_tok + _approx_tokens(txt)
            chunks.append({
                "text": txt, "heading": heading, "start_token": start_tok,
                "end_token": end_tok, "approx_tokens": _approx_tokens(txt)
            })
            # overlap
            if overlap_tokens > 0:
                tail = txt[-overlap_tokens*4:]
                buf = [tail]
                buf_tokens = _approx_tokens(tail)
                start_tok = end_tok - buf_tokens
            else:
                buf, buf_tokens = [], 0
                start_tok = end_tok

        for para in paragraphs:
            sents = _split_sentences(para)
            for s in sents:
                tok = _approx_tokens(s)
                if buf_tokens + tok > target_tokens:
                    _flush()
                buf.append(s)
                buf_tokens += tok
            if buf_tokens >= target_tokens * 0.9:
                _flush()

        _flush()  # flush remainder

    # coarse rebalancing: merge tiny trailing chunks into previous
    merged: List[Dict[str, Any]] = []
    for c in chunks:
        if merged and c["approx_tokens"] < min(200, int(0.25*target_tokens)):
            merged[-1]["text"] = (merged[-1]["text"].rstrip() + "\\n\\n" + c["text"].lstrip()).strip()
            merged[-1]["approx_tokens"] = _approx_tokens(merged[-1]["text"])
            merged[-1]["end_token"] = merged[-1]["start_token"] + merged[-1]["approx_tokens"]
        else:
            merged.append(c)
    return merged

# ---------------------------------------------------------------------
# Distillation (summary/tags)
# ---------------------------------------------------------------------
def distill_chunk(text: str) -> Dict[str, Any]:
    """
    Returns {title, summary, tags}
    """
    if not text:
        return {"title": None, "summary": "", "tags": []}
    if _oai_client is None:
        # fallback heuristic
        first_line = text.strip().split('\\n', 1)[0][:120]
        import re as _re
        words = _re.findall(r'[A-Za-z][A-Za-z0-9\\-]{3,}', text.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        tags = [w for w,_ in sorted(freq.items(), key=lambda t: (-t[1], t[0]))[:5]]
        return {"title": first_line, "summary": text[:500], "tags": tags[:MAX_TAGS]}
    prompt = (
        "Summarize the following chunk for a knowledge base.\\n"
        "Return JSON with keys: title, summary, tags (<=8 short tags).\\n"
        "Text:\\n---\\n" + text[:8000] + "\\n---"
    )
    resp = _oai_client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content": prompt}]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        first = text.strip().split('\\n', 1)[0][:120]
        return {"title": first, "summary": text[:500], "tags": []}

# ---------------------------------------------------------------------
# Dedupe helpers (exact + simhash)
# ---------------------------------------------------------------------
def _normalize_for_hash(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'\\s+', ' ', s)
    return s

def compute_dedupe_hash(s: str) -> str:
    return hashlib.sha256(_normalize_for_hash(s).encode('utf-8')).hexdigest()

def _tokenize_for_simhash(s: str) -> List[str]:
    import re as _re
    return _re.findall(r'[a-z0-9]{3,}', _normalize_for_hash(s))

def simhash64(s: str) -> int:
    import hashlib as _hashlib
    tokens = _tokenize_for_simhash(s)
    if not tokens:
        return 0
    v = [0]*NEAR_DUP_SIMHASH_BITS
    for tok in tokens:
        h = int(_hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
        for i in range(NEAR_DUP_SIMHASH_BITS):
            bit = (h >> i) & 1
            v[i] += 1 if bit else -1
    out = 0
    for i, val in enumerate(v):
        if val >= 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def is_near_duplicate(sim_a: int, sim_b: int) -> bool:
    # threshold 0.92 -> max distance <= floor(64*(1-0.92)) = 5
    max_dist = max(0, int(NEAR_DUP_SIMHASH_BITS * (1.0 - NEAR_DUP_THRESHOLD)))
    return hamming(sim_a, sim_b) <= max_dist

# ---------------------------------------------------------------------
# Ingest core
# ---------------------------------------------------------------------
def ingest_text(*, title: Optional[str], text: str, type: str, tags: Optional[List[str]] = None,
                source: Optional[str] = None, role_view: Optional[List[str]] = None,
                file_id: Optional[str] = None, session_id: Optional[str] = None,
                dedupe: bool = True) -> Dict[str, Any]:
    """
    Ingest normalized text:
    - chunk
    - distill each chunk
    - dedupe exact + near-dup (within-batch; exact dedupe also DB-level if store supports)
    - upsert memory rows + vectors
    """
    assert type in ("episodic","semantic","procedural"), "invalid type"
    norm = clean_text(text)
    chunks = chunk_text(norm)

    batch_simhashes: List[int] = []
    upserted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for ch in chunks:
        ctext = ch["text"].strip()
        if not ctext:
            continue

        # exact dedupe via sha256
        dh = compute_dedupe_hash(ctext)
        if dedupe and store is not None:
            try:
                exists = False
                if hasattr(store, "memory_exists_by_hash"):
                    exists = bool(store.memory_exists_by_hash(dh))
                elif hasattr(store, "find_memory_by_hash"):
                    exists = bool(store.find_memory_by_hash(dh))
                elif hasattr(store, "get_memory_by_hash"):
                    exists = bool(store.get_memory_by_hash(dh))
                if exists:
                    skipped.append({"reason":"duplicate","dedupe_hash": dh})
                    continue
            except Exception as ex:
                logger.warning("dedupe hash check failed: %s", ex)

        # near-dup within batch using SimHash
        sh = simhash64(ctext)
        if dedupe and batch_simhashes:
            if any(is_near_duplicate(sh, prev) for prev in batch_simhashes):
                skipped.append({"reason":"near-duplicate","simhash": sh})
                continue
        batch_simhashes.append(sh)

        # distill
        meta = distill_chunk(ctext)
        ctitle = meta.get("title") or (title or "Untitled")
        ctags = list(set((tags or []) + (meta.get("tags") or [])))[:MAX_TAGS]

        # insert memory row
        mem_row = None
        try:
            if store is None:
                mem_row = {"id": f"tmp_{len(upserted)+1}", "type": type}
            else:
                if hasattr(store, "upsert_memory"):
                    mem_row = store.upsert_memory(
                        type=type, title=ctitle, text=ctext, tags=ctags,
                        source=source or "ingest", file_id=file_id, session_id=session_id,
                        role_view=role_view or [], dedupe_hash=dh
                    )
                elif hasattr(store, "insert_memory"):
                    mem_row = store.insert_memory(
                        type=type, title=ctitle, text=ctext, tags=ctags,
                        source=source or "ingest", file_id=file_id, session_id=session_id,
                        role_view=role_view or [], dedupe_hash=dh
                    )
                else:
                    raise RuntimeError("store.upsert_memory/insert_memory not available")
        except Exception as ex:
            logger.error("Failed to insert memory row: %s", ex)
            skipped.append({"reason":"store-failed","error": str(ex)})
            continue

        mem_id = mem_row["id"]

        # entities & mentions
        ents: List[str] = []
        if _entities and hasattr(_entities, "extract_entities"):
            try:
                ents = _entities.extract_entities(ctext)
            except Exception as ex:
                logger.warning("entity extraction failed: %s", ex)

        # embedding + upsert vector
        try:
            if retrieval and hasattr(retrieval, "upsert_memory_vector"):
                retrieval.upsert_memory_vector(
                    mem_id=mem_id, user_id=None, type=type, content=ctext,
                    title=ctitle, tags=ctags, importance=4,
                    created_at_iso=None, source=source or "ingest",
                    role_view=role_view or [], entity_ids=ents or []
                )
            if store and hasattr(store, "update_memory_embedding_id"):
                try:
                    store.update_memory_embedding_id(mem_id, f"mem_{mem_id}")
                except Exception:
                    pass
        except Exception as ex:
            logger.error("vector upsert failed: %s", ex)
            skipped.append({"reason":"vector-failed","error": str(ex)})
            continue

        upserted.append({"memory_id": mem_id, "embedding_id": f"mem_{mem_id}"})

    return {"upserted": upserted, "skipped": skipped}

# ---------------------------------------------------------------------
# Convenience API matching /ingest/batch body (PRD §5.3)
def ingest_batch(items: List[Dict[str, Any]], dedupe: bool = True) -> Dict[str, Any]:
    all_up: List[Dict[str, Any]] = []
    all_sk: List[Dict[str, Any]] = []
    for it in items:
        res = ingest_text(
            title = it.get("title"),
            text = it.get("text") or "",
            type = it.get("type") or "semantic",
            tags = it.get("tags") or [],
            source = it.get("source") or "ingest",
            role_view = it.get("role_view") or [],
            file_id = it.get("file_id"),
            session_id = it.get("session_id"),
            dedupe = dedupe if it.get("dedupe", None) is None else bool(it["dedupe"]),
        )
        all_up.extend(res.get("upserted", []))
        all_sk.extend(res.get("skipped", []))
    return {"upserted": all_up, "skipped": all_sk}
