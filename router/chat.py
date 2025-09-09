# router/chat.py
import os, json, math, datetime
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index
from ingest.pipeline import normalize_text
from memory.autosave import apply_autosave
from guardrails.redteam import review_answer

router = APIRouter()
client = OpenAI()

# -------- Models --------
class ChatReq(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    role: Optional[Literal["researcher","staff","director","admin"]] = None
    preferences: Optional[Dict[str, Any]] = None
    debug: bool = False

class ChatResp(BaseModel):
    session_id: str
    answer: str
    citations: List[str]
    guidance_questions: List[str]
    autosave: Dict[str, Any]
    redteam: Dict[str, Any]
    metrics: Dict[str, Any]

# -------- Helpers --------
def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _now():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

def _load_prompt(path: str, fallback: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback

def _embed(text: str) -> List[float]:
    kwargs = {"model": os.getenv("EMBED_MODEL","text-embedding-3-small"), "input": text}
    if os.getenv("EMBED_DIM"):
        kwargs["dimensions"] = int(os.getenv("EMBED_DIM"))
    er = client.embeddings.create(**kwargs)
    return er.data[0].embedding

def _retrieve(sb, index, query: str, top_k_per_type: int = 10) -> List[Dict[str, Any]]:
    """Minimal retrieval used for /chat (reuses your Pinecone+Supabase pattern)."""
    vec = _embed(query)
    namespaces = ["semantic","episodic","procedural"]
    hits: List[Dict[str, Any]] = []
    for ns in namespaces:
        qr = index.query(vector=vec, top_k=top_k_per_type, include_metadata=True, namespace=ns)
        for m in qr.matches or []:
            mid = (m.metadata or {}).get("id") or (m.id or "").replace("mem_","")
            hits.append({
                "memory_id": mid,
                "namespace": ns,
                "score": float(m.score or 0),
            })
    # join with memories
    by_id = {}
    if hits:
        ids = list({h["memory_id"] for h in hits if h.get("memory_id")})
        rows = sb.table("memories").select("id,type,title,tags,created_at").in_("id", ids).limit(300).execute()
        data = rows.data if hasattr(rows,"data") else rows.get("data") or []
        by_id = {r["id"]: r for r in data}
    out: List[Dict[str, Any]] = []
    for h in hits:
        r = by_id.get(h["memory_id"])
        if not r: 
            continue
        out.append({
            "id": r["id"],
            "type": r["type"],
            "title": r.get("title"),
            "tags": r.get("tags") or [],
            "created_at": r.get("created_at"),
            "score": h["score"],
        })
    # simple semantic order (Phase 5 keeps it light; you already hybrid-rank in /search)
    out.sort(key=lambda x: x.get("score",0.0), reverse=True)
    return out[:12]

def _pack_context(sb, items: List[Dict[str, Any]], include_text: bool = True) -> List[Dict[str, Any]]:
    """Fetch text for top items and return [{"id","title","text"}...]"""
    if not items: 
        return []
    ids = [it["id"] for it in items]
    text_col = (os.getenv("MEMORIES_TEXT_COLUMN","value")).strip().lower()
    rows = sb.table("memories").select(f"id,title,{text_col}").in_("id", ids).limit(len(ids)).execute()
    data = rows.data if hasattr(rows,"data") else rows.get("data") or []
    by_id = {r["id"]: r for r in data}
    out = []
    for it in items:
        r = by_id.get(it["id"])
        if not r: 
            continue
        out.append({"id": it["id"], "title": r.get("title") or "", "text": r.get(text_col) or ""})
    return out

def _safe_json_answer(prompt: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call Answerer; if JSON parse fails, retry once with a stricter instruction."""
    sys = _load_prompt("prompts/system_answerer.md", "Return strict JSON.")
    # Encode context compactly
    ctx_chunks = [{"id": c["id"], "title": c["title"], "text": (c["text"][:2000])} for c in context[:8]]
    user = json.dumps({"question": prompt, "context": ctx_chunks})
    r = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL","gpt-4.1-mini"),
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0
    )
    raw = r.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        # retry with a terse reminder
        r2 = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL","gpt-4.1-mini"),
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":user},
                {"role":"system","content":"Return VALID JSON only, no prose."}
            ],
            temperature=0
        )
        return json.loads(r2.choices[0].message.content or "{}")

# -------- Route --------
@router.post("/chat", response_model=ChatResp)
def chat_chat_post(body: ChatReq, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    sb = get_client()
    index = get_index()
    t0 = datetime.datetime.utcnow()

    # 1) session
    session_id = body.session_id
    if not session_id:
        ins = sb.table("sessions").insert({"title": None}).execute()
        sel = sb.table("sessions").select("id").order("created_at", desc=True).limit(1).execute()
        session_id = (sel.data if hasattr(sel,"data") else sel.get("data") or [{}])[0].get("id")

    # 2) working memory (last few turns) — optional analytics later
    # (We still store the current turns below.)

    # 3) retrieval (lightweight inside /chat)
    retrieved_meta = _retrieve(sb, index, body.prompt, top_k_per_type=int(os.getenv("TOPK_PER_TYPE","8")))
    retrieved_chunks = _pack_context(sb, retrieved_meta, include_text=True)

    # 4) answerer (strict JSON)
    draft = _safe_json_answer(body.prompt, retrieved_chunks)
    if not isinstance(draft, dict):
        raise HTTPException(status_code=500, detail="Answerer returned non-JSON")

    # 5) reviewer
    verdict = review_answer(draft_json=draft, prompt=body.prompt, retrieved_chunks=retrieved_chunks) or {}
    action = (verdict.get("action") or "allow").lower()

    # Simple revise path: if "revise", ask the model to rewrite the answer following required_edits.
    if action == "revise":
        edits = "; ".join(verdict.get("required_edits") or [])
        sys = "Rewrite the answer applying these edits. Keep citations unchanged. Return JSON with same schema."
        user = json.dumps({"original": draft, "required_edits": edits})
        r3 = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL","gpt-4.1-mini"),
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0
        )
        try:
            draft = json.loads(r3.choices[0].message.content or "{}")
        except Exception:
            action = "block"  # fail safe

    if action == "block":
        # safe response
        safe = {
            "session_id": session_id,
            "answer": "I can’t answer confidently with the available evidence. Try adding filters or uploading the source.",
            "citations": [],
            "guidance_questions": ["Do you want me to search with a narrower tag or date range?"],
            "autosave": {"saved": False, "items": []},
            "redteam": verdict,
            "metrics": {"latency_ms": int((datetime.datetime.utcnow()-t0).total_seconds()*1000)}
        }
        return safe

    # 6) autosave
    autosave = apply_autosave(
        sb=sb,
        pinecone_index=index,
        candidates=draft.get("autosave_candidates") or [],
        session_id=session_id,
        text_col_env=os.getenv("MEMORIES_TEXT_COLUMN","value"),
    )

    # 7) persist messages
    try:
        sb.table("messages").insert({
            "session_id": session_id, "role": "user", "content": body.prompt,
            "tokens": None, "latency_ms": None, "model": os.getenv("CHAT_MODEL")
        }).execute()
        sb.table("messages").insert({
            "session_id": session_id, "role": "assistant", "content": draft.get("answer") or "",
            "tokens": None, "latency_ms": int((datetime.datetime.utcnow()-t0).total_seconds()*1000),
            "model": os.getenv("CHAT_MODEL")
        }).execute()
    except Exception:
        pass

    # 8) respond
    resp = {
        "session_id": session_id,
        "answer": draft.get("answer") or "",
        "citations": draft.get("citations") or [],
        "guidance_questions": draft.get("guidance_questions") or [],
        "autosave": autosave,
        "redteam": verdict,
        "metrics": {"latency_ms": int((datetime.datetime.utcnow()-t0).total_seconds()*1000)}
    }
    return resp
