"""
agent/ingest.py
----------------
Clean, import-safe ingest helpers used by router/upload.py:
  - clean_text(raw, mime_type=None, filename=None) -> str
  - distill_chunk(text) -> {title, summary, tags, entities}
  - ingest_text(...): persists memory, entities, vector; returns mem_id

This module avoids indentation pitfalls and tolerates missing dependencies.
"""

from __future__ import annotations

import os
import re
import time
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tolerant imports
try:
    from vendors.openai_client import client as _oai_client
    from vendors.openai_client import CHAT_MODEL as _CHAT_MODEL
except Exception:
    _oai_client = None
    _CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

try:
    from agent import store as _store
except Exception:
    _store = None

try:
    from agent import retrieval as _retrieval
except Exception:
    _retrieval = None

def _normalize_utf8(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\t\x0b\x0c]", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \u00a0]{2,}", " ", s)
    return s.strip()

# ---------------- public helpers ----------------

def clean_text(raw: Any, mime_type: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    Very conservative normalization used by /upload. If you already extracted
    text elsewhere, just pass it here to normalize whitespace.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")
    else:
        text = str(raw or "")
    return _normalize_utf8(text)

def distill_chunk(text: str) -> Dict[str, Any]:
    """
    Summarize a text chunk and extract lightweight tags/entities.
    Tries LLM; falls back to simple heuristics when LLM unavailable.
    """
    text = (text or "").strip()
    out: Dict[str, Any] = {"title": "", "summary": "", "tags": [], "entities": []}

    if not text:
        return out

    # LLM path
    if _oai_client is not None:
        try:
            prompt = (
                "You are a concise distiller. Given TEXT, return JSON with keys "
                "title, summary, tags (<=5), entities (list of {name,type} where type in "
                "['person','org','project','artifact','concept']). Keep it short."
            )
            resp = _oai_client.chat.completions.create(
                model=_CHAT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"TEXT:\n{text[:4000]}"},
                ],
            )
            cand = resp.choices[0].message.content.strip()
            # tolerate non-JSON by best-effort parse
            try:
                data = json.loads(cand)
                if isinstance(data, dict):
                    out.update({k: data.get(k, out[k]) for k in out.keys()})
                    return out
            except Exception:
                pass
        except Exception as ex:
            logger.warning("distill_chunk LLM failed: %s", ex)

    # Heuristic fallback
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    title = lines[0][:120] if lines else "Untitled"
    summary = " ".join(lines[:5])[:600]
    tags = []
    if re.search(r"meeting|minutes|decision", text, flags=re.I):
        tags.append("meeting")
    if re.search(r"procedure|SOP|checklist", text, flags=re.I):
        tags.append("sop")
    out.update({"title": title, "summary": summary, "tags": tags, "entities": []})
    return out

# ---------------- persistence entry ----------------

def ingest_text(
    *,
    title: str,
    text_payload: str,
    _type: str,
    tags: List[str],
    source: str,
    file_id: Optional[str],
    user_id: Optional[str],
    role_view: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Insert a memory row, persist entity mentions, and upsert the vector.
    Returns mem_id or None if store unavailable.
    """
    if _store is None:
        logger.error("store module not available; cannot persist ingest")
        return None

    t0 = time.time()
    meta = meta or {}
    base_tags = list({*(tags or []), *([source] if source else [])})
    text_norm = _normalize_utf8(text_payload or "")
    dedupe_hash = hashlib.sha256(text_norm.encode("utf-8")).hexdigest()

    # summarize & extract
    summary = distill_chunk(text_norm)
    if not title:
        title = summary.get("title") or "Untitled"

    # 1) write memory row
    mem_id = _store.insert_memory(
        type=_type,
        title=title,
        text=text_norm,
        tags=list({*base_tags, *(t for t in summary.get("tags", []) if isinstance(t, str))}),
        source=source,
        file_id=file_id,
        session_id=None,
        author_user_id=user_id,
        role_view=role_view,
        dedupe_hash=dedupe_hash,
    )

    # 2) upsert entities + mentions (best-effort)
    ents = summary.get("entities") or []
    for ent in ents:
        try:
            name = (ent.get("name", "") or "").strip()
            etype = (ent.get("type", "") or "").strip().lower()
            if not name or etype not in {"person", "org", "project", "artifact", "concept"}:
                continue
            e = _store.ensure_entity(name, etype)
            _store.upsert_entity_mention(e["id"], mem_id)
        except Exception as ex:
            logger.warning("entity upsert failed for %r: %s", ent, ex)

    # 3) upsert vector + write embedding id (best-effort)
    emb_id = f"mem_{mem_id}"
    try:
        if _retrieval is not None and hasattr(_retrieval, "upsert_memory_vector"):
            _retrieval.upsert_memory_vector(
                mem_id=mem_id,
                user_id=user_id,
                type=_type,
                content=text_norm,
                title=title,
                tags=list({*base_tags, *(t for t in summary.get("tags", []) if isinstance(t, str))}),
                importance=1.2,
                created_at_iso=None,
                source=source,
                role_view=role_view,
                entity_ids=[],  # could be filled with real entity ids if _store exposes them
            )
        _store.update_memory_embedding_id(mem_id, emb_id)
    except Exception as ex:
        try:
            _store.log_tool_run(
                "ingest_embed_upsert",
                {"memory_id": str(mem_id)},
                {"error": str(ex)},
                False,
                int((time.time() - t0) * 1000),
            )
        except Exception:
            pass
        logger.warning("vector upsert failed: %s", ex)

    return mem_id

__all__ = ["clean_text", "distill_chunk", "ingest_text"]
