"""
agent/pipeline.py
------------------
Sane, import-safe pipeline module with enforced red-team step.
Designed to compile cleanly and avoid "return outside function" and similar
syntax issues. Minimal external assumptions; all optional imports are gated.

Exports:
  - chat(...): returns (session_id, response_dict)

Note: This file is intentionally conservative so it doesn't break on import.
Integrate deeper with your store/retrieval modules as needed.
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -------- Optional / tolerant imports (do not fail on import) --------
try:
    # your project layout: vendors/openai_client.py should expose `client` and model constants
    from vendors.openai_client import client as _oai_client
except Exception:
    _oai_client = None

try:
    from vendors.openai_client import CHAT_MODEL as _CHAT_MODEL
except Exception:
    _CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

# agent package peers (safe to import; they should not execute code at import-time)
try:
    from agent import retrieval as _retrieval
except Exception:
    _retrieval = None

try:
    from agent import store as _store
except Exception:
    _store = None

try:
    from guardrails import redteam as _redteam_mod
except Exception:
    _redteam_mod = None

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
TOPK_PER_TYPE = int(os.getenv("TOPK_PER_TYPE", "30"))
RECENCY_HALFLIFE_DAYS = float(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))

# ------------------ Small helpers ------------------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _oai_chat(messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0) -> str:
    """
    Minimal wrapper to call OpenAI Chat Completions. If the vendors client
    isn't available, returns a fallback string.
    """
    model = model or _CHAT_MODEL
    if _oai_client is None:
        return "(LLM unavailable) No OpenAI client configured."

    try:
        resp = _oai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
        )
        # OpenAI v1-style
        return resp.choices[0].message.content.strip()
    except Exception as ex:
        logger.exception("OpenAI chat call failed: %s", ex)
        return f"(LLM error) {ex}"

def _enforce_redteam(raw_answer: str, citations: List[Dict[str, Any]], prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Call red-team reviewer and enforce allow/revise/block. Returns (answer, verdict).
    """
    verdict = {"action": "allow", "reasons": []}
    if _redteam_mod is None or not hasattr(_redteam_mod, "review"):
        return raw_answer, verdict

    try:
        vt = _redteam_mod.review(
            draft={"answer": raw_answer, "citations": citations},
            prompt=prompt,
            retrieved_chunks=retrieved_chunks,
        )
        if isinstance(vt, dict):
            verdict = vt
    except Exception as ex:
        verdict = {"action": "allow", "reasons": [f"review_error:{ex}"]}
        return raw_answer, verdict

    action = verdict.get("action", "allow")
    if action == "block":
        safe = (
            "I can’t answer confidently with the available evidence. "
            "Try narrowing by date or tag, or upload the source."
        )
        return safe, verdict
    if action == "revise":
        edits = "\n".join(verdict.get("required_edits", [])[:6])
        revised = _oai_chat(
            [
                {"role": "system", "content": "Revise the answer strictly per edits. No new claims."},
                {"role": "user", "content": f"Original:\n{raw_answer}\n\nEdits:\n{edits}"},
            ],
            model=_CHAT_MODEL,
            temperature=0.0,
        )
        return revised, verdict

    return raw_answer, verdict

# ------------------ Main entry ------------------
def chat(
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    role: Optional[str] = None,
    session_id: Optional[str] = None,
    message: str = "",
    history: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    debug: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Primary chat orchestration used by the router.
    Returns (session_id, response_dict).
    """
    t0 = _now_ms()
    history = history or []
    temperature = 0.0 if temperature is None else float(temperature)

    # 1) Ensure session id (best-effort; if store unavailable, synthesize one)
    sid = session_id
    if sid is None:
        if _store is not None and hasattr(_store, "get_or_create_session"):
            try:
                sid = _store.get_or_create_session(user_id=user_id, user_email=user_email)
            except Exception as ex:
                logger.warning("get_or_create_session failed: %s", ex)
                sid = str(uuid.uuid4())
        else:
            sid = str(uuid.uuid4())

    # 2) Retrieve candidates (optional; tolerate missing module)
    retrieved: List[Dict[str, Any]] = []
    try:
        if _retrieval is not None and hasattr(_retrieval, "retrieve"):
            retrieved = _retrieval.retrieve(
                query=message,
                role=role,
                session_id=sid,
                top_k=TOPK_PER_TYPE,
            ) or []
    except Exception as ex:
        logger.warning("retrieval failed: %s", ex)
        retrieved = []

    # 3) Build simple context and citations
    def _mk_citation(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_id": item.get("id") or item.get("memory_id"),
            "title": item.get("title") or item.get("source_title") or "memory",
            "type": item.get("type") or "semantic",
        }

    citations = [_mk_citation(x) for x in retrieved[:3]]

    # Pack context (very simple; project can replace with token-aware packer)
    context_parts: List[str] = []
    for h in history[-4:]:
        role_h = h.get("role", "user")
        content_h = h.get("content", "")
        context_parts.append(f"{role_h.upper()}: {content_h}")
    for x in retrieved[:8]:
        txt = x.get("text") or x.get("summary") or ""
        ttl = x.get("title") or "memory"
        context_parts.append(f"[{ttl}]\n{txt}")
    context_blob = "\n\n".join(context_parts).strip()

    # 4) Ask the LLM
    system = "You are SUAPS Brain. Be concise and specific. Ground answers in SUAPS data when provided."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{message}\n\nContext:\n{context_blob}"},
    ]
    raw_answer = _oai_chat(messages, model=_CHAT_MODEL, temperature=temperature)

    # 5) Red-team (enforce)
    raw_answer, verdict = _enforce_redteam(raw_answer, citations, message, retrieved)

    # 6) Guidance questions (seed 1–2)
    guidance_questions = [
        "Does this align with the current publication deliverables?",
        "Should we capture a checklist as procedural memory?",
    ][:2]

    # 7) Autosave stub (leave decisions to autosave module if present)
    autosave = {"saved": False, "items": []}

    # 8) Metrics
    latency_ms = _now_ms() - t0
    metrics = {"latency_ms": latency_ms}

    # 9) Persist messages (best-effort)
    try:
        if _store is not None and hasattr(_store, "save_message"):
            _store.save_message(session_id=sid, role="user", content=message)
            _store.save_message(session_id=sid, role="assistant", content=raw_answer)
    except Exception as ex:
        logger.warning("saving messages failed: %s", ex)

    # 10) Build response
    response = {
        "session_id": sid,
        "answer": raw_answer,
        "citations": citations,
        "guidance_questions": guidance_questions,
        "autosave": autosave,
        "redteam": verdict,
        "metrics": metrics,
    }
    return sid, response
