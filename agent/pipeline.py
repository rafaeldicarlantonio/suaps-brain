from __future__ import annotations

import os, time, json, logging
from typing import Any, Dict, List, Optional, Tuple

from vendors.openai_client import client, CHAT_MODEL, EMBED_MODEL
from vendors.pinecone_client import get_index
from vendors.supabase_client import supabase

from agent import store
from validators.json import strict_parse_or_retry
from memory.selection import rank_and_pack_minimal

logger = logging.getLogger(__name__)

ANSWER_SYS = (
    "You are SUAPS Brain. Be concise and specific. Ground answers in SUAPS data. "
    "Return ONLY valid JSON with keys: answer (string), citations (array of strings), "
    "guidance_questions (array of strings), autosave_candidates (array of objects with "
    "fact_type, title, text, tags, confidence)."
)

def _intent(prompt: str) -> str:
    p = prompt.lower()
    if "upload" in p or "ingest" in p or "attach" in p:
        return "ingest"
    if "health" in p or "debug" in p:
        return "admin"
    return "qa"

def _recency_decay(days: float, half_life: int = int(os.getenv("RECENCY_HALFLIFE_DAYS","90"))) -> float:
    import math
    return math.exp(-math.log(2) * (days / float(half_life)))

def _embed(text: str) -> List[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def _pine_query(query: str, namespace: str, top_k: int = 10) -> List[Dict[str,Any]]:
    idx = get_index()
    vec = _embed(query)
    res = idx.query(vector=vec, top_k=top_k, namespace=namespace, include_metadata=True)
    # unify shape
    matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
    out=[]
    for m in matches or []:
        meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        out.append({
            "id": m.get("id") if isinstance(m, dict) else getattr(m, "id", None),
            "score": m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0),
            "namespace": namespace,
            "metadata": meta,
        })
    return out

def _fetch_memories(ids: List[str]) -> List[Dict[str,Any]]:
    if not ids:
        return []
    # ids are like mem_<uuid>; strip mem_ to look up embedding_id or store both
    r = supabase.table("memories").select("*").in_("embedding_id", ids).execute()
    return r.data or []

def _answer_llm(context_blocks: List[Dict[str,Any]], prompt: str) -> Dict[str, Any]:
    # Build context text
    ctx = []
    for c in context_blocks:
        title = c.get("title") or ""
        txt = c.get("text") or ""
        mid = c.get("id") or c.get("memory_id") or ""
        ctx.append(f"[{mid}] {title}\n{txt}")
    sys_prompt = ANSWER_SYS
    user = (
        "Use only the provided context. "
        "Cite memory_ids you used. "
        "If evidence is weak, say so. "
        f"PROMPT:\n{prompt}\n---\nCONTEXT:\n" + "\n\n".join(ctx[:10])
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user}],
    )
    content = resp.choices[0].message.content
    try:
        obj = strict_parse_or_retry(content, None)
        return obj
    except Exception:
        # one retry: ask to produce JSON again
        user2 = user + "\nReturn ONLY valid JSON per schema."
        resp2 = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user2}],
        )
        return strict_parse_or_retry(resp2.choices[0].message.content, None)

def handle_chat(req: Dict[str,Any]) -> Dict[str,Any]:
    t0 = time.time()
    prompt = req.get("prompt","").strip()
    session_id = req.get("session_id")
    role = req.get("role")
    session = store.ensure_session(session_id=session_id, title=None)

    # 1) working memory (last 2 turns)
    wm = store.fetch_recent_messages(session["id"], limit=4)

    # 2) intent
    intent = _intent(prompt)
    if intent != "qa":
        return {"session_id": session["id"], "answer": "This endpoint currently supports QA only.", "citations": [], "guidance_questions": [], "autosave": {"saved": False}, "redteam": {"action":"allow","reasons":["non-qa request"]}, "metrics":{"latency_ms": int((time.time()-t0)*1000)}}

    # 3) per-type vector search (minimal day1)
    hits = []
    for ns in ("episodic","semantic","procedural"):
        try:
            hits.extend(_pine_query(prompt, namespace=ns, top_k=int(os.getenv("TOPK_PER_TYPE","10"))))
        except Exception as ex:
            logger.warning(f"pinecone query failed for {ns}: {ex}")

    # 4) fetch records
    embedding_ids = [h["id"] for h in hits if h.get("id")]
    recs = _fetch_memories(embedding_ids)

    # 5) rank + pack (day1 minimal)
    packed = rank_and_pack_minimal(hits, recs, wm, prompt)

    # 6) answer
    draft = _answer_llm(packed["context"], prompt)

    # 7) red-team (day1 stub: allow)
    redteam = {"action":"allow","reasons":[]}

    # 8) autosave (day1: tolerate absence of candidates)
    autosave = {"saved": False, "items": []}
    try:
        from memory.autosave import autosave_from_candidates
        autosave = autosave_from_candidates(draft.get("autosave_candidates",[]), session["id"])
    except Exception as ex:
        logger.warning(f"autosave failed: {ex}")

    # 9) persist messages best-effort
    try:
        store.insert_message(session["id"], "user", prompt, model=None, tokens=None, latency_ms=None)
        store.insert_message(session["id"], "assistant", draft.get("answer",""), model=CHAT_MODEL, tokens=None, latency_ms=int((time.time()-t0)*1000))
    except Exception as ex:
        logger.warning(f"persist messages failed: {ex}")

    return {
        "session_id": session["id"],
        "answer": draft.get("answer",""),
        "citations": draft.get("citations",[]),
        "guidance_questions": draft.get("guidance_questions",[]),
        "autosave": autosave,
        "redteam": redteam,
        "metrics": {"latency_ms": int((time.time()-t0)*1000)},
    }
