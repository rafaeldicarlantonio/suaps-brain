# agent/pipeline.py
import time
import os
from typing import List, Dict, Tuple, Optional, Any

from vendors.openai_client import client as openai_client, CHAT_MODEL
from agent import store, retrieval

# ranking/packing helpers (we added these files earlier)
from memory.selection import hybrid_rank, pack_to_budget, one_hop_graph

# red-team (allow graceful fallback if not present)
try:
    from guardrails import redteam
except Exception:
    redteam = None  # type: ignore
# extractor model can be smaller than main chat model
EXTRACTOR_MODEL = os.getenv("OPENAI_EXTRACTOR_MODEL", CHAT_MODEL)

def _extract_autosave_candidates(answer: str) -> list[dict]:
    """
    Call LLM to extract high-signal facts for autosave.
    Output: list of {fact_type, title, text, tags, confidence}
    """
    try:
        prompt = (
            "Extract important decisions, deadlines, and procedures from the following text. "
            "Return a JSON array where each item has: "
            "fact_type (decision|deadline|procedure|entity), title, text, tags (array), confidence (0-1). "
            f"\n\nText:\n{answer}"
        )
        resp = openai_client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "system", "content": "You are a JSON-only extractor."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        import json
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception as ex:
        # fail gracefully
        return []

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
DEFAULT_CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------
def _ensure_session(user_id: str, session_id: Optional[str]) -> str:
    """Return an existing session_id or create a new one using store.ensure_session/create_session."""
    if session_id:
        return session_id

    # Prefer DB-backed
    try:
        if hasattr(store, "ensure_session"):
            s = store.ensure_session(session_id, user_id)  # type: ignore
            # expected to return {"id": "..."}:
            if isinstance(s, dict) and "id" in s:
                return s["id"]
    except Exception:
        pass

    try:
        if hasattr(store, "create_session"):
            s = store.create_session(user_id)  # type: ignore
            if isinstance(s, dict) and "id" in s:
                return s["id"]
            if isinstance(s, str):
                return s
    except Exception:
        pass

    # Fallback: ephemeral
    try:
        import ulid
        return str(ulid.new())
    except Exception:
        import uuid
        return f"sess_{uuid.uuid4()}"

def _trim_history(history: List[Dict], limit: int = MAX_HISTORY_TURNS) -> List[Dict]:
    if not history or limit <= 0:
        return []
    return history[-limit:]

def _save_message(session_id: str, role: str, content: str) -> None:
    """Best-effort persistence; ignore if store lacks these helpers."""
    try:
        if hasattr(store, "insert_message"):
            store.insert_message(session_id, role, content)  # type: ignore
        elif hasattr(store, "log_message"):
            store.log_message(session_id, role=role, content=content)  # type: ignore
    except Exception:
        pass

def _llm_answer(message: str, context_blocks: List[Dict[str, Any]], temperature: Optional[float]) -> Any:
    # Build prompt (simple & robust)
    sys = (
        "You are SUAPS Brain. Be concise and specific. "
        "Ground answers in retrieved context if provided. "
        "If evidence is weak or missing, say so and propose next steps."
    )

    ctx_txt = ""
    if context_blocks:
        lines = []
        for i, m in enumerate(context_blocks, 1):
            title = (m.get("title") or "")[:120]
            snippet = (m.get("text") or m.get("content") or "")[:1200]
            mid = m.get("memory_id") or m.get("id") or ""
            t = m.get("type") or m.get("__namespace") or ""
            lines.append(f"[{i}] ({t}) {title}\n{snippet}\n(id:{mid})")
        ctx_txt = "Retrieved context:\n\n" + "\n\n---\n\n".join(lines)

    msgs = [{"role": "system", "content": sys}]
    if ctx_txt:
        msgs.append({"role": "system", "content": ctx_txt})
    msgs.append({"role": "user", "content": message})

    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        temperature=DEFAULT_CHAT_TEMPERATURE if temperature is None else float(temperature),
    )
    return resp

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def chat(
    user_id: str,
    session_id: Optional[str],
    message: str,
    history: List[Dict],
    temperature: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (session_id, draft_dict)
    draft_dict has:
      answer: str
      citations: List[str]
      guidance_questions: List[str]
      autosave_candidates: List[dict]
      redteam: dict
      metrics: dict
    """
    t0 = time.time()

    # 1) Ensure session and log user message
    sid = _ensure_session(user_id, session_id)
    _save_message(sid, "user", message)

    # 2) Retrieval
    # Use our unified per-type search that already exists in agent/retrieval.py
    try:
        hits = retrieval.search_per_type(user_id, message)  # List[metadata dicts]
    except Exception:
        hits = []

    # 3) Graph expand (1-hop via entity mentions)
    try:
        expand = one_hop_graph(hits, db=store.supabase, limit=10)  # may return raw rows
    except Exception:
        expand = []

    # Normalize raw rows to metadata-like dicts (best effort)
    norm_expand = []
    for r in expand:
        if isinstance(r, dict):
            d = dict(r)
            if "id" in d and "memory_id" not in d:
                d["memory_id"] = d["id"]
            norm_expand.append(d)
    candidates = hits + norm_expand

    # 4) Hybrid rank + pack
    try:
        ranked = hybrid_rank(message, candidates, q_entities=set())
    except Exception:
        ranked = candidates

    # Token budget packing (history is already a list of turns; we only pass ranked as context)
    try:
        context = pack_to_budget(history=[], ranked=ranked)
    except Exception:
        context = ranked

    # 5) Primary LLM answer
    resp = _llm_answer(message, context, temperature)
    raw_answer = resp.choices[0].message.content if resp and resp.choices else ""

    # 6) Build draft (PRD shape). We keep it simple: citations from top ranked memory_ids.
    citations = []
    for md in ranked[:3]:
        mid = md.get("memory_id") or md.get("id")
        if mid:
            citations.append(str(mid))

    autosave_candidates = _extract_autosave_candidates(raw_answer)

draft: Dict[str, Any] = {
    "answer": raw_answer,
    "citations": citations,
    "guidance_questions": [],
    "autosave_candidates": autosave_candidates,
    "metrics": {
        "tokens": getattr(resp, "usage", None).total_tokens if getattr(resp, "usage", None) else None,
        "latency_ms": int((time.time() - t0) * 1000),
    },
}


    # 7) Red-team reviewer (best-effort)
    try:
        if redteam is not None and hasattr(redteam, "review"):
            verdict = redteam.review(message, raw_answer, ranked)
        else:
            verdict = {"action": "allow", "reasons": []}
    except Exception as ex:
        verdict = {"action": "allow", "error": str(ex)}
    draft["redteam"] = verdict

    # 8) Save assistant message (best-effort)
    _save_message(sid, "assistant", draft["answer"])

    return sid, draft
