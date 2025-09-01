# agent/ingest.py
# PRD-aligned ingestion: normalize -> chunk -> summarize -> dedupe (sha256 + SimHash) -> store -> embed -> graph
from __future__ import annotations
import os, re, time, json, math, hashlib
from typing import Dict, List, Optional, Tuple
from vendors.openai_client import client, CHAT_MODEL, EMBED_MODEL
from agent import store
from agent.retrieval import upsert_memory_vector
from vendors.supabase_client import supabase

# Try import simhash; if missing, near-dup check becomes no-op (exact dedupe still enforced)
try:
    from simhash import Simhash
    HAVE_SIMHASH = True
except Exception:
    HAVE_SIMHASH = False

# ------------- normalization -------------
def clean_text(s: str) -> str:
    if not s:
        return ""
    # normalize newlines and spaces
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[\t\x0b\x0c]', ' ', s)
    # drop common page headers/footers (very heuristic)
    s = re.sub(r'\n\s*Page \d+\s*\n', '\n', s, flags=re.IGNORECASE)
    # collapse >2 newlines
    s = re.sub(r'\n{3,}', '\n\n', s)
    # trim trailing spaces on lines
    s = '\n'.join([ln.strip() for ln in s.split('\n')])
    return s.strip()

# ------------- chunking -------------
TARGET_TOKENS = int(os.getenv("INGEST_TARGET_TOKENS", "1000"))  # ~800–1200
OVERLAP_TOKENS = int(os.getenv("INGEST_OVERLAP_TOKENS", "120"))

def _approx_tokens(s: str) -> int:
    # cheap approx: 1 token ~= 4 chars
    return max(1, math.ceil(len(s) / 4))

def _split_sections(s: str) -> List[str]:
    # split on markdown headings first
    parts = re.split(r'(?m)^(?=\s*#)', s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [s]

def _split_paragraphs(s: str) -> List[str]:
    paras = [p.strip() for p in s.split('\n\n') if p.strip()]
    return paras if paras else [s]

def chunk_text(s: str) -> List[str]:
    sections = _split_sections(s)
    chunks: List[str] = []
    budget = TARGET_TOKENS
    overlap = OVERLAP_TOKENS
    for sec in sections:
        paras = _split_paragraphs(sec)
        cur = []
        cur_tokens = 0
        for p in paras:
            tks = _approx_tokens(p)
            if cur_tokens + tks > budget and cur:
                chunks.append('\n\n'.join(cur))
                # start next window with overlap from previous
                ov = []
                # take tail of previous chunk to overlap ~OVERLAP_TOKENS
                back = '\n\n'.join(cur)
                # naive slice by chars approximating tokens*4
                ov_chars = OVERLAP_TOKENS * 4
                ov_text = back[-ov_chars:]
                cur = [ov_text, p]
                cur_tokens = _approx_tokens(ov_text) + tks
            else:
                cur.append(p)
                cur_tokens += tks
        if cur:
            chunks.append('\n\n'.join(cur))
    return [c for c in chunks if c.strip()]

# ------------- summarize + tag + entities -------------
SUMMARIZE_SYS = "You extract structured metadata for knowledge chunks. Be concise and factual. Return strict JSON."
def summarize_chunk(text: str) -> Dict[str, any]:
    prompt = {
        "instructions": "Summarize and tag the chunk. Extract entity candidates (name,type in person|org|project|artifact|concept).",
        "chunk": text[:8000]
    }
    resp = client.chat.completions.create(
        model=os.getenv("EXTRACTOR_MODEL", CHAT_MODEL),
        temperature=0,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":SUMMARIZE_SYS},
            {"role":"user","content":json.dumps(prompt)}
        ]
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}
    title = (data.get("title") or "").strip() or (text.strip().split("\n",1)[0][:120] if text.strip() else "Untitled")
    summary = (data.get("summary") or "").strip()
    tags = data.get("tags") or []
    ents = data.get("entities") or []
    # normalize entities
    norm_ents = []
    for e in ents:
        if isinstance(e, dict):
            name = (e.get("name") or "").strip()
            typ = (e.get("type") or "").strip().lower()
            if name and typ:
                norm_ents.append({"name": name, "type": typ})
    return {"title": title, "summary": summary, "tags": tags, "entities": norm_ents}

# ------------- dedupe -------------
def _simhash64(s: str) -> Optional[int]:
    if not HAVE_SIMHASH:
        return None
    try:
        return Simhash(s).value  # 64-bit int
    except Exception:
        return None

def _simhash_sim(a: int, b: int) -> float:
    # similarity = 1 - HammingDistance/64
    x = a ^ b
    hd = 0
    while x:
        x &= x - 1
        hd += 1
    return 1.0 - (hd / 64.0)

NEAR_DUP_THRESHOLD = float(os.getenv("NEAR_DUP_THRESHOLD", "0.92"))

# ------------- main distill -------------
def distill_chunk(*, user_id: Optional[str], raw_text: str, base_tags: List[str] | None = None, make_qa: bool = False,
                  type: str = "semantic", role_view: Optional[List[str]] = None, source: str = "upload",
                  file_id: Optional[str] = None) -> List[str]:
    t0 = time.time()
    base_tags = base_tags or []
    s = clean_text(raw_text)
    chunks = chunk_text(s)
    inserted_ids: List[str] = []
    # Preload recent memory texts for near-dup simhash
    recent = store.fetch_recent_memories_texts(limit=500) if HAVE_SIMHASH else []
    recent_hashes = [(r["id"], store.sha256_normalized((r.get("text") or ""))) for r in recent]
    recent_sims = [(r["id"], Simhash(r.get("text") or "").value) for r in recent] if HAVE_SIMHASH else []

    for ch in chunks:
        meta = summarize_chunk(ch)
        title = meta["title"]
        # combine: summary + original chunk for embedding richness
        text_payload = meta["summary"].strip() + "\n\n" + ch.strip() if meta["summary"] else ch.strip()
        dedupe_hash = store.sha256_normalized(text_payload)

        # exact dedupe
        if store.find_memory_by_dedupe_hash(dedupe_hash):
            store.log_tool_run("ingest_dedupe_exact", {"title": title}, {"skipped":"duplicate"}, True, int((time.time()-t0)*1000))
            continue

        # near-dup (SimHash)
        if HAVE_SIMHASH:
            try:
                h = Simhash(text_payload).value
                is_near = False
                for (_id, s64) in recent_sims:
                    if s64 is None: 
                        continue
                    sim = _simhash_sim(h, s64)
                    if sim >= NEAR_DUP_THRESHOLD:
                        is_near = True
                        break
                if is_near:
                    store.log_tool_run("ingest_dedupe_near", {"title": title}, {"skipped":"near-duplicate","threshold":NEAR_DUP_THRESHOLD}, True, int((time.time()-t0)*1000))
                    continue
            except Exception:
                pass

        # insert memory row
        tags = list({*base_tags, *[t for t in meta["tags"] if isinstance(t,str)]})
        mem_id = store.insert_memory(type=type, title=title, text=text_payload, tags=tags, source=source,
                                     file_id=file_id, session_id=None, author_user_id=user_id, role_view=role_view, dedupe_hash=dedupe_hash)

        # upsert to pinecone (embedding_id is vector id)
        try:
            upsert_memory_vector(mem_id=mem_id, user_id=user_id, type_=type, content=text_payload, title=title,
                                 tags=tags, importance=1, created_at_iso=None, source=source, role_view=role_view, entity_ids=[])
            emb_id = f"mem_{mem_id}"
            store.update_memory_embedding_id(mem_id, emb_id)
        except Exception as ex:
            store.log_tool_run("ingest_embed_upsert", {"memory_id": mem_id}, {"error": str(ex)}, False, int((time.time()-t0)*1000))

        # entities
        try:
            for e in meta["entities"]:
                eid = store.get_or_create_entity(e["name"], e["type"])
                store.link_entity_to_memory(eid, mem_id, weight=1.0)
        except Exception as ex:
            store.log_tool_run("ingest_entities", {"memory_id": mem_id}, {"error": str(ex)}, False, int((time.time()-t0)*1000))

        inserted_ids.append(mem_id)

    store.log_tool_run("ingest_pipeline", {"chunks": len(chunks)}, {"inserted": len(inserted_ids)}, True, int((time.time()-t0)*1000))
    return inserted_ids
