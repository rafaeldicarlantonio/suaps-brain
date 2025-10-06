# router/chat.py
import os
import json
import datetime
from uuid import uuid4
from typing import Optional, List, Dict, Any, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator
from openai import OpenAI

from vendors.supabase_client import get_client
from vendors.pinecone_client import get_index, safe_query
from ingest.pipeline import normalize_text
from memory.autosave import apply_autosave
from guardrails.redteam import review_answer
from auth.light_identity import ensure_user  # <-- attribution helper
from memory.graph import expand_entities
from extractors.signals import extract_signals_from_text  # <-- NEW: fallback extractor

router = APIRouter()
client = OpenAI()


# ---------- Models ----------
class ChatReq(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None
    role: Optional[Literal["researcher", "staff", "director", "admin"]] = None
    preferences: Optional[Dict[str, Any]] = None
    debug: bool = False

    # allow unknown fields instead of 422'ing
    model_config = ConfigDict(extra="ignore")

    # automatically create a prompt from messages[] if none is provided
    @model_validator(mode="before")
    def ensure_prompt(cls, values):
        if isinstance(values, dict) and not values.get("prompt"):
            msgs = values.get("messages") or []
            user_bits = [m.get("content", "") for m in msgs if m.get("role") == "user"]
            if user_bits:
                values["prompt"] = " ".join(user_bits)[:4000]
        if isinstance(values, dict) and not values.get("prompt"):
            raise ValueError("prompt or messages is required")
        return values

class ChatResp(BaseModel):
    session_id: str
    answer: str
    citations: List[str]
    guidance_questions: List[str]
    autosave: Dict[str, Any]
    redteam: Dict[str, Any]
    metrics: Dict[str, Any]


# ---------- Helpers ----------
def _auth(x_api_key: Optional[str]):
    expected = os.getenv("X_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _embed(text: str) -> List[float]:
    kwargs: Dict[str, Any] = {"model": os.getenv("EMBED_MODEL", "text-embedding-3-small"), "input": text}
    dim = os.getenv("EMBED_DIM")
    if dim:
        kwargs["dimensions"] = int(dim)
    er = client.embeddings.create(**kwargs)
    return er.data[0].embedding


def _retrieve(sb, index, query: str, top_k_per_type: int = 8) -> List[Dict[str, Any]]:
    vec = _embed(query)
    namespaces = ["semantic", "episodic", "procedural"]
    hits: List[Dict[str, Any]] = []

    for ns in namespaces:
        res = safe_query(index, vector=vec, top_k=top_k_per_type, include_metadata=True, namespace=ns)
        for m in res.matches:
            md = m.metadata or {}
            mem_id = (md.get("id") or (m.id or "")).replace("mem_", "")
            if not mem_id:
                continue
            hits.append({"memory_id": mem_id, "namespace": ns, "score": float(m.score or 0.0)})

    ids = list({h["memory_id"] for h in hits})
    by_id: Dict[str, Dict[str, Any]] = {}
    if ids:
        rows = (
            sb.table("memories")
            .select("id,type,title,tags,created_at")
            .in_("id", ids)
            .limit(len(ids))
            .execute()
        )
        data = rows.data if hasattr(rows, "data") else rows.get("data") or []
        by_id = {r["id"]: r for r in data}

    out: List[Dict[str, Any]] = []
    for h in hits:
        r = by_id.get(h["memory_id"])
        if not r:
            continue
        out.append(
            {
                "id": r["id"],
                "type": r["type"],
                "title": r.get("title"),
                "tags": r.get("tags") or [],
                "created_at": r.get("created_at"),
                "score": h["score"],
            }
        )

    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out[:12]


def _pack_context(sb, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []

    ids = [it["id"] for it in items]
    text_col = (os.getenv("MEMORIES_TEXT_COLUMN", "text")).strip().lower()

    # Only fetch what actually exists in 'memories'
    rows = sb.table("memories").select(f"id,title,type,{text_col}") \
              .in_("id", ids).limit(len(ids)).execute()

    data = rows.data if hasattr(rows, "data") else rows.get("data") or []
    by_id = {r["id"]: r for r in data}

    out: List[Dict[str, Any]] = []
    for it in items:
        r = by_id.get(it["id"])
        if not r:
            continue

        mem_type = (r.get("type") or "semantic").upper()
        raw_text = r.get(text_col) or ""
        norm_text = normalize_text(raw_text)

        # Label the text with its memory type
        labeled_text = f"[{mem_type} MEMORY] {norm_text}"

        out.append({
            "id": it["id"],
            "title": r.get("title") or "",
            "text": labeled_text,
            "type": mem_type
        })

    return out


def _answer_json(prompt: str, context_str: str) -> Dict[str, Any]:
    sys = """You are SUAPS Brain. Be concise and specific. Mentor tone: strategic, supportive.
    Always ground answers in SUAPS data. Cite the memory IDs you used.

    You will see different types of memory in context:
    - [SEMANTIC MEMORY]: definitions, background knowledge.
    - [EPISODIC MEMORY]: time-stamped events, meetings, decisions.
    - [PROCEDURAL MEMORY]: rules, SOPs, how-to steps.

    Use each type appropriately: semantic for explanations, episodic for timelines, procedural for rules.

    Return STRICT JSON only with this schema:
    {...}
    """

    user = json.dumps({"question": prompt, "context": context_str})
    r = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        temperature=0,
    )
    raw = r.choices[0].message.content or "{}"
    return json.loads(raw)



# ---------- Route ----------
@router.post("/chat", response_model=ChatResp)
def chat_chat_post(
    body: ChatReq,
    x_api_key: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None),  # attribution header
):
    _auth(x_api_key)
    sb = get_client()
    index = get_index()
    t0 = datetime.datetime.utcnow()

    # Resolve/ensure user (best-effort)
    author_user_id = ensure_user(sb=sb, email=x_user_email)

    # Ensure a session id
    session_id = body.session_id
    if not session_id:
        # Try DB-generated id
        payload = {"title": None}
        if author_user_id:
            payload["user_id"] = author_user_id
        try:
            sb.table("sessions").insert(payload).execute()
            sel = sb.table("sessions").select("id").order("created_at", desc=True).limit(1).execute()
            data = sel.data if hasattr(sel, "data") else sel.get("data") or []
            if data:
                session_id = data[0]["id"]
        except Exception:
            # Local fallback
            session_id = str(uuid4())

    # Retrieval
    retrieved_meta = _retrieve(
        sb,
        index,
        body.prompt,
        top_k_per_type=int(os.getenv("TOPK_PER_TYPE", "8"))
    )
    retrieved_chunks = _pack_context(sb, retrieved_meta)

    # 🔗 Graph Expansion (3 hops) - non-fatal
    try:
        graph_neighbors = expand_entities(sb, retrieved_chunks, max_hops=3, max_neighbors=10, max_per_entity=3)
        retrieved_chunks.extend(graph_neighbors)
    except Exception as e:
        # non-fatal: log or ignore if graph expansion fails
        print("Graph expansion failed:", e)

    # Build context string with memory-type labels
    context_for_llm = "\n".join(chunk["text"] for chunk in retrieved_chunks)

    # Collect just the IDs (for schema validation of "citations")
    context_ids = [chunk["id"] for chunk in retrieved_chunks]

    # Answer
    draft = _answer_json(body.prompt, context_for_llm)
    if not isinstance(draft, dict):
        raise HTTPException(status_code=500, detail="Answerer returned non-JSON")

    # Ensure citations are always a list of strings (ids only)
    if "citations" in draft and isinstance(draft["citations"], list):
        draft["citations"] = [
            c if isinstance(c, str) else c.get("id") for c in draft["citations"]
        ]

    # Red-team (non-fatal)
    try:
        verdict = review_answer(
            draft_json=draft,
            prompt=body.prompt,
            retrieved_chunks=retrieved_chunks
        ) or {}
    except Exception:
        verdict = {"action": "allow", "reasons": []}
    action = (verdict.get("action") or "allow").lower()

    if action == "block":
        return {
            "session_id": session_id,
            "answer": "I can’t answer confidently with the available evidence. Try adding filters or uploading the source.",
            "citations": [],
            "guidance_questions": ["Do you want me to search with a narrower tag or date range?"],
            "autosave": {"saved": False, "items": []},
            "redteam": verdict,
            "metrics": {"latency_ms": int((datetime.datetime.utcnow() - t0).total_seconds() * 1000)},
        }

    # -----------------------
    # Autosave (non-fatal) with robust fallback
    # -----------------------
    try:
        # Prefer LLM-provided autosave candidates
        candidates = (draft.get("autosave_candidates") or []).copy()

        # If none, derive from USER + ASSISTANT + small CONTEXT sample
        if not candidates:
            sample_ctx = "\n\n".join(
                [(c.get("text") or "")[:1200] for c in (retrieved_chunks[:2] if retrieved_chunks else [])]
            )
            fallback_text = (
                f"USER:\n{(body.prompt or '')[:4000]}\n\n"
                f"ASSISTANT:\n{(draft.get('answer') or '')[:4000]}\n\n"
                f"CONTEXT:\n{sample_ctx}"
            )
            derived = extract_signals_from_text(fallback_text) or []

            # Tag derived items with provenance + session for traceability
            for d in derived:
                tags = set(d.get("tags") or [])
                tags.update({"source:chat", f"session:{session_id}"})
                d["tags"] = sorted(list(tags))
            candidates.extend(derived)

        autosave = apply_autosave(
            sb=sb,
            pinecone_index=index,
            candidates=candidates,
            session_id=session_id,
            text_col_env=os.getenv("MEMORIES_TEXT_COLUMN", "text"),
            author_user_id=author_user_id,  # pass attribution
        )
    except Exception:
        autosave = {"saved": False, "items": []}

    # Persist messages (best-effort)
    try:
        sb.table("messages").insert(
            {"session_id": session_id, "role": "user", "content": body.prompt, "model": os.getenv("CHAT_MODEL")}
        ).execute()
        sb.table("messages").insert(
            {
                "session_id": session_id,
                "role": "assistant",
                "content": draft.get("answer") or "",
                "model": os.getenv("CHAT_MODEL"),
                "latency_ms": int((datetime.datetime.utcnow() - t0).total_seconds() * 1000),
            }
        ).execute()
    except Exception:
        pass

    return {
        "session_id": session_id,
        "answer": draft.get("answer") or "",
        "citations": draft.get("citations") or [],
        "guidance_questions": draft.get("guidance_questions") or [],
        "autosave": autosave,
        "redteam": verdict,
        "metrics": {"latency_ms": int((datetime.datetime.utcnow() - t0).total_seconds() * 1000)},
    }
