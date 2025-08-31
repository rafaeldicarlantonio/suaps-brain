# agent/pipeline.py
import os
from typing import List, Dict, Tuple, Optional

from vendors.openai_client import client
from agent import store

# Try to import retrieval; if not present or missing functions, we will degrade gracefully.
try:
    from agent import retrieval  # type: ignore
except Exception:
    retrieval = None  # type: ignore

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Default chat temperature when the caller doesn't provide one
DEFAULT_CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.5"))

# If your upstream rejects the temperature parameter entirely, set to "false"
CHAT_ALLOW_TEMPERATURE = os.getenv("CHAT_ALLOW_TEMPERATURE", "true").lower() == "true"

# Token budget knobs (coarse; we do simple trimming by count of turns)
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

# Retrieval knobs
K_EPISODIC = int(os.getenv("K_EPI", "3"))
K_SEMANTIC = int(os.getenv("K_SEM", "5"))

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _ensure_session(user_id: str, session_id: Optional[str]) -> str:
    """
    Return an existing session_id or create a new one via store.create_session(user_id).
    If create_session isn't available, generate a simple ULID-like string.
    """
    if session_id:
        return session_id
    # Prefer DB-backed session if the store provides it
    try:
        if hasattr(store, "create_session"):
            s = store.create_session(user_id)  # expected to return {"id": "..."} or similar
            if isinstance(s, dict) and "id" in s:
                return s["id"]
            if isinstance(s, str):
                return s
    except Exception:
        pass

    # Fallback: generate a simple session token
    try:
        import ulid
        return str(ulid.new())
    except Exception:
        import uuid
        return f"sess_{uuid.uuid4()}"

def _trim_history(history: List[Dict], limit: int = MAX_HISTORY_TURNS) -> List[Dict]:
    """
    Keep only the last N turns. History is expected as a list of {role, content}.
    """
    if not history or limit <= 0:
        return []
    return history[-limit:]

def _get_procedural_rules() -> str:
    """
    System prompt (procedural memory). Adjust as needed.
    """
    return (
        "You are SUAPS Brain — the Society for UAP Studies' mentor and institutional memory. "
        "Be concise, precise, and grounded in retrieved context. "
        "If unsure, say so and suggest next steps. "
        "When appropriate, ask one clarifying or guiding question that advances the user's goal. "
        "Avoid hallucinations; cite retrieved snippets when you rely on them."
    )

def _fetch_context(user_id: str, query: str) -> List[Dict]:
    """
    Try multiple retrieval functions if available; return a list of dicts with
    keys: {'type','title','content','score','tags'} (best-effort).
    If retrieval is not configured or fails, return [].
    """
    results: List[Dict] = []

    if retrieval is None:
        return results

    # We don't know the exact retrieval API in your repo;
    # try several common patterns and soft-fail if missing.
    try:
        # Pattern A: a consolidated fetch_context
        if hasattr(retrieval, "fetch_context"):
            ctx = retrieval.fetch_context(user_id=user_id, query=query, k_ep=K_EPISODIC, k_sem=K_SEMANTIC)
            if isinstance(ctx, list):
                return ctx
    except Exception:
        pass

    try:
        # Pattern B: separate search functions
        episodic_hits = []
        semantic_hits = []

        if hasattr(retrieval, "search_episodic"):
            episodic_hits = retrieval.search_episodic(user_id=user_id, query=query, k=K_EPISODIC) or []
        if hasattr(retrieval, "search_semantic"):
            semantic_hits = retrieval.search_semantic(query=query, k=K_SEMANTIC) or []

        # Normalize structure
        def norm(hit, tlabel):
            if isinstance(hit, dict):
                return {
                    "type": tlabel,
                    "title": hit.get("title") or "",
                    "content": hit.get("content") or hit.get("text") or "",
                    "score": hit.get("score"),
                    "tags": hit.get("tags") or [],
                }
            return {"type": tlabel, "title": "", "content": str(hit), "score": None, "tags": []}

        results.extend([norm(h, "episodic") for h in episodic_hits])
        results.extend([norm(h, "semantic") for h in semantic_hits])
        return results
    except Exception:
        pass

    return results

def _assemble_messages(
    procedural: str,
    context_snippets: List[Dict],
    history: List[Dict],
    user_message: str,
) -> List[Dict]:
    """
    Build OpenAI Chat API messages list.
    """
    msgs: List[Dict] = []
    msgs.append({"role": "system", "content": procedural})

    # Add retrieved context as one assistant message to keep the prompt compact
    if context_snippets:
        ctx_lines = []
        for i, snip in enumerate(context_snippets, start=1):
            title = snip.get("title") or ""
            body = snip.get("content") or ""
            ttype = snip.get("type") or ""
            tags = snip.get("tags") or []
            line = f"[{i}] ({ttype}) {title}\n{body}"
            if tags:
                line += f"\nTags: {', '.join([str(t) for t in tags])}"
            ctx_lines.append(line)
        ctx_block = "Retrieved context:\n\n" + "\n\n---\n\n".join(ctx_lines)
        msgs.append({"role": "system", "content": ctx_block})

    # Prior history
    for turn in history:
        r = turn.get("role")
        c = turn.get("content")
        if r in ("user", "assistant", "system") and isinstance(c, str):
            msgs.append({"role": r, "content": c})

    # Current user message
    msgs.append({"role": "user", "content": user_message})
    return msgs

def _call_llm(messages: List[Dict], temperature: Optional[float]) -> str:
    """
    Call the provider. Add temperature only if allowed. Return text content.
    """
    kwargs = {"model": CHAT_MODEL, "messages": messages}
    final_temp = DEFAULT_CHAT_TEMPERATURE if temperature is None else float(temperature)
    if CHAT_ALLOW_TEMPERATURE:
        kwargs["temperature"] = final_temp

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""

# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------
def chat(
    user_id: str,
    session_id: Optional[str],
    message: str,
    history: List[Dict],
    temperature: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Main chat entrypoint used by app.py.
    Returns: (session_id, answer)
    """
    sid = _ensure_session(user_id, session_id)
    trimmed_history = _trim_history(history, MAX_HISTORY_TURNS)

    # Gather context (best-effort; ok if empty)
    ctx = _fetch_context(user_id=user_id, query=message)

    # Build prompt
    procedural = _get_procedural_rules()
    messages = _assemble_messages(
        procedural=procedural,
        context_snippets=ctx,
        history=trimmed_history,
        user_message=message,
    )

    # LLM call
    answer = _call_llm(messages, temperature=temperature)

    # Optionally: log the turn to a messages table, if store provides it
    try:
        if hasattr(store, "log_message"):
            store.log_message(sid, role="user", content=message)
            store.log_message(sid, role="assistant", content=answer)
    except Exception:
        pass

    return sid, answer
