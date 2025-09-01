# agent/pipeline.py
import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any

from vendors.openai_client import client as openai_client, CHAT_MODEL
from agent import store, retrieval
from memory.selection import hybrid_rank, pack_to_budget, one_hop_graph

# red-team (graceful fallback)
try:
    from guardrails import redteam
except Exception:
    redteam = None  # type: ignore

# extractor model can be smaller than main chat model
EXTRACTOR_MODEL = os.getenv("OPENAI_EXTRACTOR_MODEL", CHAT_MODEL)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
DEFAULT_CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _ensure_session(user_id: str, session_id: Optional[str]) -> str:
    """Return an existing session_id or create a new one via store.create_session/ensure_session."""
    if session_id:
        return session_id
    try:
        if hasattr(store, "create_session"):
            s = store.create_session(user_id)  # expected to return {"id": "..."} or str
            if isinstance(s, dict) and "id" in s:
                return s["id"]
            if isinstance(s, str):
                return s
    except Exception:
        pass
    try:
        import ulid
        return str(ulid.new())
    except Exception:
        return f"sess_{uuid.uuid4()}"

def _trim_history(history: List[Dict], limit: int = MAX_HISTORY_TURNS) -> List[Dict]:
    if not history or limit <= 0:
        return []
    return history[-limit:]

def _save_message(session_id: str, role: str, content: str) -> None:
    try:
        if hasattr(store, "insert_message"):
            store.insert_message(session_id, role, content)  # type: ignore
        elif hasattr(store, "log_message"):
            store.log_message(session_id, role=role, content=content)  # type: ignore
    except Exception:
        pass

def _get_procedural_rules() -> str:
    return (
        "You are SUAPS Brain — the Society for UAP Studies' mentor and institutional memory. "
        "Be concise, precise, and grounded in retrieved context. "
        "If unsure, say so and suggest next steps. "
        "When appropriate, ask one clarifying or guiding question that advances the user's goal. "
        "Avoid hallucinations; cite retrieved snippets when you rely on them."
    )

def _llm_answer(messages: List[Dict], temperature: Optional[float]) -> Any:
    kwargs = {"model": CHAT_MODEL, "messages": messages}
    final_temp = DEFAULT_CHAT_TEMPERATURE if temperature is None else float(temperature)
    if os.getenv("CHAT_ALLOW_TEMPERATURE", "true").lower() == "true":
        kwargs["temperature"] = final_temp
    return openai_client.chat.completions.create(**kwargs)

def _extract_autosave_candidates(answer: str) -> list[dict]:
    """
    Use a small LLM to extract structured autosave candidates from the answer.
    """
    try:
        prompt = (
            "Extract important decisions, deadlines, and procedures from the following text. "
            "Return a JSON array where each item has: "
            "fact_type (decision|deadline|procedure|entity), title, text, tags (array), confidence (0-1). "
            f"\\n\\nText:\\n{answer}"
        )
        resp = openai_client.chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-only extractor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        import json
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

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
    Main chat entrypoint used by app.py.
    Returns: (session_id, draft_dict)
    """
    t0 = time.time()
    sid = _ensure_session(user_id, session_id)
    trimmed_history = _trim_history(history, MAX_HISTORY_TURNS)

    # Retrieval
    try:
        hits = retrieval.search_per_type(user_id, message)
    except Exception:
        hits = []

    try:
        expand = one_hop_graph(hits, db=store.supabase, limit=10)
    except Exception:
        expand = []

    norm_expand = []
    for r in expand:
        if isinstance(r, dict):
            d = dict(r)
            if "id" in d and "memory_id" not in d:
                d["memory_id"] = d["id"]
            norm_expand.append(d)

    candidates = hits + norm_expand

    try:
        ranked = hybrid_rank(message, candidates, q_entities=set())
    except Exception:
        ranked = candidates

    try:
        context = pack_to_budget(history=[], ranked=ranked)
    except Exception:
        context = ranked

    # LLM call
    resp = _llm_answer(
        _assemble_messages(_get_procedural_rules(), context, trimmed_history, message),
        temperature,
    )
    raw_answer = resp.choices[0].message.content if resp and resp.choices else ""

    # Build draft
    citations = [str(md.get("memory_id") or md.get("id")) for md in ranked[:3] if md.get("memory_id") or md.get("id")]
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

    # Red-team review (graceful)
    try:
        if redteam and hasattr(redteam, "review"):
            verdict = redteam.review(message, raw_answer, ranked)
        else:
            verdict = {"action": "allow", "reasons": []}
    except Exception as ex:
        verdict = {"action": "allow", "error": str(ex)}
    draft["redteam"] = verdict

    # Save assistant message
    _save_message(sid, "assistant", draft["answer"])

    return sid, draft
