
"""
agent/pipeline.py
-----------------
PRD-compliant /chat orchestration:
- Retrieval per type (role-aware)
- Hybrid ranking + 1-hop graph (uses memory.selection if available)
- LLM answer in STRICT PRD JSON schema
- Red-team reviewer with allow/revise/block
- Autosave from chat (policy thresholds)
- Message persistence (best-effort)
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Imports (tolerant; module may not exist in some environments)
# ---------------------------------------------------------------------
try:
    from vendors.openai_client import client as _oai_client
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", CHAT_MODEL)
except Exception:
    _oai_client = None
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
    EXTRACTOR_MODEL = CHAT_MODEL

try:
    from agent import store
except Exception:
    store = None

try:
    from agent import retrieval
except Exception:
    retrieval = None

# selection helpers (hybrid rank + pack + graph); provide fallbacks if missing
try:
    from memory import selection as _sel
except Exception:
    _sel = None

try:
    from memory import autosave as _autosave
except Exception:
    _autosave = None

try:
    from guardrails import redteam as _redteam
except Exception:
    _redteam = None

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
TOPK_PER_TYPE      = int(os.getenv("TOPK_PER_TYPE", "30"))
RECENCY_HALFLIFE_D = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))
AUTOSAVE_CONF_THRESHOLD = float(os.getenv("AUTOSAVE_CONF_THRESHOLD", "0.75"))
AUTOSAVE_ENTITY_CONF_THRESHOLD = float(os.getenv("AUTOSAVE_ENTITY_CONF_THRESHOLD", "0.85"))

# ---------------------------------------------------------------------
# Utility: cheap tokenizer (avoid external deps)
# ---------------------------------------------------------------------
def _approx_tokens(s: str) -> int:
    # 1 token ~ 4 chars heuristic
    return max(1, int(len(s or "") / 4))

# ---------------------------------------------------------------------
# Selection fallbacks if memory.selection is absent
# ---------------------------------------------------------------------
def _fallback_hybrid_rank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Very simple deterministic scoring: Pinecone score primary.
    scored = []
    for c in candidates:
        s = float(c.get("score", 0.0))
        scored.append((s, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]

def _fallback_pack_to_budget(history: List[Dict[str, Any]], ranked: List[Dict[str, Any]], budget_tokens: int) -> Dict[str, Any]:
    # Pack last 2 user/assistant turns + top ranked chunks (text+title) until budget
    context_parts: List[str] = []
    # last 2 turns if provided
    if history:
        recent = history[-4:]
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content") or turn.get("text") or ""
            context_parts.append(f"[{role}] {content}".strip())
    used = _approx_tokens("\n".join(context_parts))
    for r in ranked:
        if used >= budget_tokens:
            break
        txt = r.get("text") or ""
        title = r.get("title") or ""
        snippet = (title + "\n" + txt).strip()
        t = _approx_tokens(snippet)
        if used + t > budget_tokens:
            continue
        context_parts.append(f"[{r.get('type','?')}:{r.get('id')}] {snippet}")
        used += t
    return {"context": "\n\n".join(context_parts)}

# ---------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------
ANSWERER_SYSTEM_PROMPT = (
    "You are SUAPS Brain. Be concise and specific. Mentor tone: strategic, supportive.\n"
    "Always ground answers in SUAPS data. Cite memory_ids used.\n"
    "If evidence is weak or missing, say so and propose next steps.\n"
    "Return STRICT JSON only that matches the schema provided."
)

ANSWER_SCHEMA_JSON = {
    "type":"object",
    "properties":{
        "answer":{"type":"string"},
        "citations":{"type":"array","items":{"type":"string"}},
        "guidance_questions":{"type":"array","items":{"type":"string"}},
        "autosave_candidates":{"type":"array","items":{
            "type":"object",
            "properties":{
                "fact_type":{"type":"string","enum":["decision","deadline","procedure","entity"]},
                "title":{"type":"string"},
                "text":{"type":"string"},
                "tags":{"type":"array","items":{"type":"string"}},
                "confidence":{"type":"number","minimum":0,"maximum":1}
            },
            "required":["fact_type","title","text","confidence"]
        }}
    },
    "required":["answer","citations","guidance_questions","autosave_candidates"]
}
from validators.json import strict_parse_or_retry

try:
    draft_json = strict_parse_or_retry(llm_resp.choices[0].message.content)
except Exception:
    # Single repair attempt with response_format
    llm_resp = _oai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages_for_answerer,
    )
    draft_json = strict_parse_or_retry(llm_resp.choices[0].message.content)

def _llm_json(prompt: str, context: str, temperature: Optional[float]=None) -> Dict[str, Any]:
    if _oai_client is None:
        raise RuntimeError("OpenAI client unavailable")
    sys_msg = {"role":"system","content": ANSWERER_SYSTEM_PROMPT + "\n\nJSON schema:\n" + json.dumps(ANSWER_SCHEMA_JSON)}
    usr_msg = {"role":"user","content": f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nReturn JSON only."}
    t = 0.2 if temperature is None else temperature
    resp = _oai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=t,
        messages=[sys_msg, usr_msg],
        response_format={"type":"json_object"}
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        # one repair pass
        repair_msg = {"role":"user","content": f"The previous output was not valid JSON. Fix and return JSON only. Here is the invalid output:\n\n{raw}"}
        resp2 = _oai_client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            messages=[sys_msg, repair_msg],
            response_format={"type":"json_object"}
        )
        raw2 = resp2.choices[0].message.content
        return json.loads(raw2)

# ---------------------------------------------------------------------
# Red-team wrapper (accepts various signatures)
# ---------------------------------------------------------------------
def _review_with_redteam(draft: Dict[str, Any], prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    default_allow = {"action": "allow", "reasons": []}
    if _redteam is None:
        return default_allow
    try:
        if hasattr(_redteam, "review"):
            try:
                return _redteam.review(draft=draft, prompt=prompt, retrieved_chunks=retrieved_chunks)
            except TypeError:
                # fallback to older signature: (message, raw_answer, ranked)
                return _redteam.review(prompt, draft.get("answer",""), retrieved_chunks)
    except Exception as ex:
        logger.warning("redteam review failed: %s", ex)
    return default_allow

def _apply_required_edits(draft: Dict[str, Any], verdict: Dict[str, Any]) -> Dict[str, Any]:
    edits = verdict.get("required_edits") or []
    if not edits:
        return draft
    if _oai_client is None:
        # naive local patch: append note
        draft["answer"] = draft.get("answer","") + "\n\n[Edits applied: " + "; ".join(edits) + "]"
        return draft
    sys_msg = {"role":"system","content":"Revise the JSON object to satisfy the list of required edits. Return the FULL JSON object only."}
    usr = {
        "role":"user",
        "content": f"JSON:\n{json.dumps(draft)}\n\nRequired edits:\n{json.dumps(edits)}"
    }
    resp = _oai_client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.1,
        messages=[sys_msg, usr],
        response_format={"type":"json_object"}
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return draft

# ---------------------------------------------------------------------
# Autosave
# ---------------------------------------------------------------------
def _autosave_from_candidates(cands: List[Dict[str, Any]], session_id: Optional[str]) -> Dict[str, Any]:
    result = {"saved": False, "items": []}
    if not cands:
        return result
    try:
        if _autosave and hasattr(_autosave, "save_from_candidates"):
            return _autosave.save_from_candidates(
                candidates=cands,
                session_id=session_id,
                conf_threshold=AUTOSAVE_CONF_THRESHOLD,
                entity_conf_threshold=AUTOSAVE_ENTITY_CONF_THRESHOLD,
            )
    except Exception as ex:
        logger.warning("autosave module failed: %s", ex)

    # Fallback inline saver (minimal): save only decision/deadline/procedure above threshold
    if store is None or retrieval is None:
        return result
    saved = []
    for c in cands:
        ft = c.get("fact_type")
        conf = float(c.get("confidence", 0))
        if ft not in ("decision","deadline","procedure"):
            continue
        if conf < AUTOSAVE_CONF_THRESHOLD:
            continue
        title = (c.get("title") or ft).strip()
        text = (c.get("text") or "").strip()
        tags = c.get("tags") or []
        if not text:
            continue
        # dedupe
        dedupe_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            row = store.upsert_memory_from_chat(
                type= "episodic" if ft in ("decision","deadline") else "procedural",
                title= title,
                text= text,
                tags= tags,
                session_id=session_id,
                dedupe_hash=dedupe_hash,
            )
            retrieval.upsert_memory_vector(
                mem_id=row["id"], user_id=None, type=row.get("type","episodic"),
                content=text, title=title, tags=tags, importance=4,
                created_at_iso=None, source="chat", role_view=None, entity_ids=[]
            )
            saved.append({"memory_id": row["id"], "type": row.get("type")})
        except Exception as ex:
            logger.warning("fallback autosave failed: %s", ex)
    result["saved"] = bool(saved)
    result["items"] = saved
    return result

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _resolve_user(user_id: Optional[str], user_email: Optional[str]) -> Dict[str, Any]:
    if store is None:
        return {"id": user_id or "00000000-0000-0000-0000-000000000000", "email": user_email}
    try:
        return store.ensure_user(user_id, user_email)
    except Exception:
        return {"id": user_id or "00000000-0000-0000-0000-000000000000", "email": user_email}

def _persist_message(session_id: str, role: str, content: str, tokens: int, model: str, latency_ms: int) -> None:
    if store is None:
        return
    try:
        if hasattr(store, "insert_message"):
            store.insert_message(session_id, role, content, tokens, latency_ms, model)
        elif hasattr(store, "save_message"):
            store.save_message(session_id, role, content, tokens, latency_ms, model)
    except Exception as ex:
        logger.warning("persist message failed: %s", ex)

def _resolve_citations(ids: List[str], id2rec: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i in ids:
        r = id2rec.get(i) or {}
        out.append({
            "memory_id": i,
            "title": r.get("title"),
            "type": r.get("type"),
            "tags": r.get("tags") or [],
            "source": r.get("source"),
        })
    return out

# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def chat(*, user_id: Optional[str], user_email: Optional[str], role: Optional[str], session_id: Optional[str], message: str, history: Optional[List[Dict[str, Any]]] = None, temperature: Optional[float] = None, debug: bool = False) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    if not message or not message.strip():
        raise ValueError("message required")

    # 0) Resolve session & user
    if store and hasattr(store, "get_or_create_session"):
        sid = store.get_or_create_session(session_id, user_id, user_email)
    else:
        sid = session_id or "00000000-0000-0000-0000-000000000001"
    _ = _resolve_user(user_id, user_email)

    # 1) Working memory (recent turns) - best effort
    wm = (history or [])

    # 2) Intent (simple rule per PRD; you can expand later)
    text_l = message.lower()
    if any(k in text_l for k in ["upload","attach","ingest"]):
        # handled by other endpoints; answer with guidance
        result = {
            "session_id": sid,
            "answer": "This looks like an ingest request. Use /upload or /ingest/batch.",
            "citations": [],
            "guidance_questions": ["Do you want me to ingest a file or paste the text?"],
            "autosave": {"saved": False, "items": []},
            "redteam": {"action": "allow", "reasons": []},
            "metrics": {"tokens": 0, "latency_ms": int((time.time()-t0)*1000)}
        }
        return sid, result
    if any(k in text_l for k in ["debug","health"]):
        result = {
            "session_id": sid,
            "answer": "Use /debug/selftest or /healthz for diagnostics.",
            "citations": [],
            "guidance_questions": ["Is there a specific subsystem you want to test (OpenAI, Pinecone, Supabase)?"],
            "autosave": {"saved": False, "items": []},
            "redteam": {"action": "allow", "reasons": []},
            "metrics": {"tokens": 0, "latency_ms": int((time.time()-t0)*1000)}
        }
        return sid, result

    # 3) Retrieval per type with role-aware filter
    candidates: List[Dict[str, Any]] = []
    if retrieval is None or not hasattr(retrieval, "retrieve"):
        logger.warning("retrieval.retrieve unavailable; answering from working memory only")
    else:
        try:
            candidates = retrieval.retrieve(
                query=message,
                role=role,
                session_id=sid,
                top_k=TOPK_PER_TYPE,
                types=["episodic","semantic","procedural"],
                tags_any=None,
            )
        except Exception as ex:
            logger.warning("retrieval failed: %s", ex)
            candidates = []

    # 4) Fetch records + graph expand
    id2rec: Dict[str, Dict[str, Any]] = {r.get("id"): r for r in candidates if r.get("id")}
    records = list(id2rec.values())
    expanded: List[Dict[str, Any]] = []
    if _sel and hasattr(_sel, "one_hop_graph"):
        try:
            expanded = _sel.one_hop_graph(records, limit=10)
        except Exception as ex:
            logger.warning("one_hop_graph failed: %s", ex)

    merged_candidates = []
    seen = set()
    for r in (records + expanded):
        rid = r.get("id")
        if rid and rid not in seen:
            merged_candidates.append(r)
            seen.add(rid)

    # 5) Rank + pack
    if _sel and hasattr(_sel, "hybrid_rank"):
        try:
            ranked = _sel.hybrid_rank(merged_candidates, message)
        except Exception as ex:
            logger.warning("hybrid_rank failed, using fallback: %s", ex)
            ranked = _fallback_hybrid_rank(message, merged_candidates)
    else:
        ranked = _fallback_hybrid_rank(message, merged_candidates)

    if _sel and hasattr(_sel, "pack_to_budget"):
        try:
            packed = _sel.pack_to_budget(wm, ranked, MAX_CONTEXT_TOKENS)
        except Exception as ex:
            logger.warning("pack_to_budget failed, using fallback: %s", ex)
            packed = _fallback_pack_to_budget(wm, ranked, MAX_CONTEXT_TOKENS)
    else:
        packed = _fallback_pack_to_budget(wm, ranked, MAX_CONTEXT_TOKENS)

    context_blob = packed.get("context","")

    # 6) Primary answer (strict JSON)
    draft = _llm_json(message, context_blob, temperature=temperature)

    # 7) Red-team reviewer (pre-answer filter)
    verdict = _review_with_redteam(draft, message, ranked)
    if not isinstance(verdict, dict):
        verdict = {"action":"allow","reasons":[]}
    action = verdict.get("action","allow")
    if action == "revise":
        draft = _apply_required_edits(draft, verdict)
    elif action == "block":
        result = {
            "session_id": sid,
            "answer": "I can’t answer confidently with the available evidence. Try filters (date/tag) or upload the source.",
            "citations": [],
            "guidance_questions": ["Do you have a specific date range or tag to narrow this down?"],
            "autosave": {"saved": False, "items": []},
            "redteam": {"action": "block", "reasons": verdict.get("reasons", [])},
            "metrics": {"tokens": 0, "latency_ms": int((time.time()-t0)*1000)}
        }
        # Persist messages (best-effort)
        _persist_message(sid, "user", message, _approx_tokens(message), CHAT_MODEL, 0)
        _persist_message(sid, "assistant", result["answer"], _approx_tokens(result["answer"]), CHAT_MODEL, result["metrics"]["latency_ms"])
        return sid, result

    # 8) Autosave
    autosave_report = _autosave_from_candidates(draft.get("autosave_candidates") or [], session_id=sid)

    # 9) Persist messages (best-effort)
    latency_ms = int((time.time()-t0)*1000)
    _persist_message(sid, "user", message, _approx_tokens(message), CHAT_MODEL, 0)
    _persist_message(sid, "assistant", draft.get("answer",""), _approx_tokens(draft.get("answer","")), CHAT_MODEL, latency_ms)

    # 10) Build citations (resolve memory_id -> title/tags/type)
    cids = [c for c in draft.get("citations") or [] if isinstance(c, str)]
    citations = _resolve_citations(cids, id2rec)

    # 11) Respond
    result = {
        "session_id": sid,
        "answer": draft.get("answer",""),
        "citations": citations,
        "guidance_questions": list(draft.get("guidance_questions") or [])[:2],
        "autosave": autosave_report,
        "redteam": {"action": verdict.get("action","allow"), "reasons": verdict.get("reasons", [])},
        "metrics": {"tokens": 0, "latency_ms": latency_ms}
    }
    return sid, result
