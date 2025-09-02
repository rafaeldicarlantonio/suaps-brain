# agent/pipeline.py
import os
import time
import uuid
from typing import List, Dict, Tuple, Optional, Any

from vendors.openai_client import client as openai_client, CHAT_MODEL
from agent import store, retrieval
from memory.selection import hybrid_rank, pack_to_budget, one_hop_graph

try:
    from guardrails import redteam
except Exception:
    redteam = None  # type: ignore

EXTRACTOR_MODEL = os.getenv("OPENAI_EXTRACTOR_MODEL", CHAT_MODEL)
DEFAULT_CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "8"))

# -------------------------
# Helpers
# -------------------------
def _ensure_session(user_id: str, session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    try:
        if hasattr(store, "create_session"):
            s = store.create_session(user_id)
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
            store.insert_message(session_id, role, content)
        elif hasattr(store, "log_message"):
            store.log_message(session_id, role=role, content=content)
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

def _assemble_messages(procedural: str, context_blocks: List[Dict[str, Any]], history: List[Dict], user_message: str) -> List[Dict]:
    msgs: List[Dict] = [{"role": "system", "content": procedural}]
    if context_blocks:
        lines = []
        for i, m in enumerate(context_blocks, 1):
            title = (m.get("title") or "")[:120]
            snippet = (m.get("text") or m.get("content") or "")[:1200]
            mid = m.get("memory_id") or m.get("id") or ""
            t = m.get("type") or m.get("__namespace") or ""
            lines.append(f"[{i}] ({t}) {title}\n{snippet}\n(id:{mid})")
        ctx_txt = "Retrieved context:\n\n" + "\n\n---\n\n".join(lines)
        msgs.append({"role": "system", "content": ctx_txt})
    for turn in history:
        r, c = turn.get("role"), turn.get("content")
        if r in ("user","assistant","system") and isinstance(c, str):
            msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": user_message})
    return msgs

def _llm_answer(messages: List[Dict], temperature: Optional[float]) -> Any:
    kwargs = {"model": _MODEL, "messages": messages}
    final_temp = DEFAULT__TEMPERATURE if temperature is None else float(temperature)
    if os.getenv("_ALLOW_TEMPERATURE", "true").lower() == "true":
        kwargs["temperature"] = final_temp
    return openai_client..completions.create(**kwargs)

def _extract_autosave_candidates(answer: str) -> list[dict]:
    try:
        prompt = (
            "Extract important decisions, deadlines, and procedures from the following text. "
            "Return a JSON array where each item has: "
            "fact_type (decision|deadline|procedure|entity), title, text, tags (array), confidence (0-1). "
            f"\n\nText:\n{answer}"
        )
        resp = openai_client..completions.create(
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
        return data if isinstance(data, list) else []
    except Exception:
        return []

# -------------------------
# Public API
# -------------------------
def chat(
    user_id: str,
    session_id: Optional[str],
    message: str,
    history: List[Dict],
    temperature: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    sid = _ensure_session(user_id, session_id)
    trimmed_history = _trim_history(history, MAX_HISTORY_TURNS)

    # Retrieval
    try:
        hits = retrieval.search_per_type(user_id, message)
    except Exception:
        hits = []

    # Graph expand
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

    # Rank + pack
    try:
        ranked = hybrid_rank(message, candidates, q_entities=set())
    except Exception:
        ranked = candidates

    try:
        context = pack_to_budget(history=[], ranked=ranked)
    except Exception:
        context = ranked

    # LLM
    resp = _llm_answer(_assemble_messages(_get_procedural_rules(), context, trimmed_history, message), temperature)
    raw_answer = resp.choices[0].message.content if resp and resp.choices else ""

    # Draft
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
# --- Red-team (enforce) ---
verdict = {"action": "allow", "reasons": []}

try:
    if redteam and hasattr(redteam, "review"):
        verdict = redteam.review(
            draft={"answer": raw_answer, "citations": citations},
            prompt=message,
            retrieved_chunks=ranked,
        ) or {"action": "allow", "reasons": []}
except Exception as ex:
    verdict = {"action": "allow", "reasons": [f"review_error:{ex}"]}

# expose verdict in your response/draft if you keep that object
draft["redteam"] = verdict

if verdict["action"] == "block":
    raw_answer = (
        "I can’t answer confidently with the available evidence. "
        "Try narrowing by date or tag, or upload the source."
    )
elif verdict["action"] == "revise":
    edit_instructions = "\n".join(verdict.get("required_edits", [])[:6])
    try:
        revised = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Revise the answer strictly per edits. No new claims."},
                {"role": "user", "content": f"Original:\n{raw_answer}\n\nEdits:\n{edit_instructions}"},
            ],
        )
        raw_answer = revised.choices[0].message.content.strip()
    except Exception as ex:
        verdict["reasons"].append(f"revise_error:{ex}")
# --- end red-team (enforce) ---

   
    # Save assistant turn
    _save_message(sid, "assistant", draft["answer"])

    return sid, draft
