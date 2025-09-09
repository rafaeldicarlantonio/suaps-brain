# ingest/pipeline.py
import os
import re
import hashlib
import datetime
from typing import List, Dict, Any, Optional

from ingest.simhash import simhash64, hamming  # expects your existing file

# -----------------------------
# small utilities
# -----------------------------
def now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(s: str, chunk_size: int, overlap: int) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    chunks: List[str] = []
    i, n = 0, len(s)
    step = max(1, chunk_size - overlap)
    max_chunks = int(os.getenv("MAX_CHUNKS_PER_FILE", "500"))
    while i < n and len(chunks) < max_chunks:
        chunks.append(s[i : i + chunk_size])
        i += step
    return chunks

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# -----------------------------
# SimHash unsigned<->signed converters (BIGINT safe)
# -----------------------------
_U64 = 1 << 64
_S64 = 1 << 63

def u64_to_signed(u: int) -> int:
    """Map unsigned 64-bit 0..2^64-1 into signed BIGINT -2^63..2^63-1."""
    return u - _U64 if u >= _S64 else u

def signed_to_u64(s: int) -> int:
    """Map signed BIGINT -2^63..2^63-1 back to unsigned 64-bit."""
    return s + _U64 if s < 0 else s

# -----------------------------
# LLM helpers (best-effort; fail closed)
# -----------------------------
def llm_chunk_meta(text: str) -> Dict[str, Any]:
    """
    Return {"title": str|None, "summary": str|None, "tags": List[str]}.
    On any error, return empty fields.
    """
    try:
        from openai import OpenAI
        oai = OpenAI()
        sys = "You are an expert technical summarizer. Return strict JSON {title, summary, tags}."
        prompt = (
            f"Text:\n{text[:4000]}\n\n"
            "Return JSON with concise title (<=120 chars), 1–3 sentence summary (<=1000 chars), "
            "and 2–5 short tags."
        )
        r = oai.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        import json
        data = json.loads(r.choices[0].message.content or "{}")
        return {
            "title": (str(data.get("title") or "")[:120] or None),
            "summary": (str(data.get("summary") or "")[:1000] or None),
            "tags": [t.strip()[:32] for t in (data.get("tags") or []) if isinstance(t, str)][:6],
        }
    except Exception:
        return {"title": None, "summary": None, "tags": []}

def llm_entities(text: str) -> List[Dict[str, str]]:
    """
    Return [{'name': str, 'type': 'person'|'org'|'project'|'artifact'|'concept'}] (max ~20).
    Fail closed to [] on error.
    """
    try:
        from openai import OpenAI
        oai = OpenAI()
        sys = (
            "Extract named entities as a JSON array with items {name, type} where type is one of "
            "[person, org, project, artifact, concept]. Return JSON only."
        )
        msg = f"Text:\n{text[:3000]}"
        r = oai.chat.completions.create(
            model=os.getenv("EXTRACTOR_MODEL", os.getenv("CHAT_MODEL", "gpt-4.1-mini")),
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": msg}],
            temperature=0,
        )
        import json
        data = json.loads(r.choices[0].message.content or "[]")
        out: List[Dict[str, str]] = []
        for d in data:
            name = (d.get("name") or "").strip()
            typ = (d.get("type") or "").strip().lower()
            if name and typ in {"person", "org", "project", "artifact", "concept"}:
                out.append({"name": name[:120], "type": typ})
        return out[:20]
    except Exception:
        return []

# -----------------------------
# Entities in DB
# -----------------------------
def upsert_entity(sb, name: str, typ: str) -> Optional[str]:
    try:
        sel = sb.table("entities").select("id").eq("name", name).eq("type", typ).limit(1).execute()
        rows = sel.data if hasattr(sel, "data") else sel.get("data")
        if rows:
            return rows[0]["id"]
        sb.table("entities").insert({"name": name, "type": typ}).execute()
        sel2 = sb.table("entities").select("id").eq("name", name).eq("type", typ).limit(1).execute()
        rows2 = sel2.data if hasattr(sel2, "data") else sel2.get("data")
        return rows2[0]["id"] if rows2 else None
    except Exception:
        return None

def link_entities(sb, memory_id: str, ents: List[Dict[str, str]]) -> List[str]:
    ids: List[str] = []
    for e in ents:
        eid = upsert_entity(sb, e["name"], e["type"])
        if not eid:
            continue
        ids.append(eid)
        try:
            sb.table("entity_mentions").insert(
                {"entity_id": eid, "memory_id": memory_id, "weight": 1.0}
            ).execute()
        except Exception:
            pass
    return ids

# -----------------------------
# Main upsert pipeline (used by upload/ingest and autosave)
# -----------------------------
def upsert_memories_from_chunks(
    *,
    sb,
    pinecone_index,
    embedder,  # kept for signature compatibility (unused)
    file_id: Optional[str],
    title_prefix: str,
    chunks: List[str],
    mem_type: str = "semantic",
    tags: Optional[List[str]] = None,
    role_view: Optional[List[str]] = None,
    source: str = "upload",
    text_col_env: str = "value",
) -> Dict[str, Any]:
    """
    - exact duplicates (sha256) are skipped
    - near-duplicates (SimHash Hamming <= SIMHASH_DISTANCE) are updated in-place when UPSERT_MODE=update
      or appended as new when UPSERT_MODE=append
    - entity_ids are linked and included in Pinecone metadata
    """
    tags = tags or []
    role_view = role_view or []
    text_col = (os.getenv("MEMORIES_TEXT_COLUMN") or text_col_env).strip().lower()
    mode = (os.getenv("UPSERT_MODE", "update")).lower()
    sim_thresh = int(os.getenv("SIMHASH_DISTANCE", "6"))

    created: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    updated: List[Dict[str, Any]] = []

    def embed(text: str) -> Optional[List[float]]:
        try:
            from openai import OpenAI
            oai = OpenAI()
            kwargs = {"model": os.getenv("EMBED_MODEL", "text-embedding-3-small"), "input": text}
            if os.getenv("EMBED_DIM"):
                kwargs["dimensions"] = int(os.getenv("EMBED_DIM"))
            er = oai.embeddings.create(**kwargs)
            return er.data[0].embedding
        except Exception:
            return None

    for idx, raw in enumerate(chunks):
        text = normalize_text(raw)
        if not text:
            skipped.append({"idx": idx, "reason": "empty"})
            continue

        dedupe_hash = sha256_hex(text)
        sh_u = simhash64(text)        # unsigned 64-bit
        sh_s = u64_to_signed(sh_u)    # signed BIGINT-safe

        # ---- exact duplicate
        existing = (
            sb.table("memories")
            .select("id,embedding_id")
            .eq("dedupe_hash", dedupe_hash)
            .limit(1)
            .execute()
        )
        rows = existing.data if hasattr(existing, "data") else existing.get("data")
        if rows:
            skipped.append({"idx": idx, "reason": "duplicate", "memory_id": rows[0]["id"]})
            continue

        # ---- near-duplicate search (recent, same type)
        near = (
            sb.table("memories")
            .select(f"id,{text_col},simhash64,created_at")
            .eq("type", mem_type)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        near_rows = near.data if hasattr(near, "data") else near.get("data") or []
        nearest = None
        best_hd = 65
        for r in near_rows:
            sim = r.get("simhash64")
            if sim is None:
                continue
            sim_u = signed_to_u64(int(sim))  # DB -> unsigned
            hd = hamming(sim_u, sh_u)
            if hd < best_hd:
                best_hd, nearest = hd, r

        # ---- LLM metadata
        meta = llm_chunk_meta(text)
        title = meta["title"] or f"{title_prefix} — part {idx + 1}"
        summary = meta["summary"]
        tagset = list({*(tags or []), *meta["tags"]})

        # ---- update-in-place for near-dup
        if nearest and best_hd <= sim_thresh and mode == "update":
            try:
                sb.table("memories").update(
                    {
                        text_col: text,
                        "dedupe_hash": dedupe_hash,
                        "simhash64": sh_s,  # signed
                        "title": title,
                        "summary": summary,
                        "tags": tagset,
                        "updated_at": datetime.datetime.utcnow().isoformat(),
                        "file_id": file_id,
                        "source": source,
                    }
                ).eq("id", nearest["id"]).execute()
                memory_id = nearest["id"]

                # embed + upsert vector
                vec = embed(text)
                if vec:
                    vector_id = f"mem_{memory_id}"
                    namespace = {"semantic": "semantic", "episodic": "episodic", "procedural": "procedural"}[mem_type]
                    eid_list = link_entities(sb, memory_id, llm_entities(text))
                    metadata = {
                        "type": mem_type,
                        "title": title,
                        "tags": tagset,
                        "created_at": now_iso(),
                        "role_view": role_view,
                        "entity_ids": eid_list,
                        "source": source,
                    }
                    pinecone_index.upsert(
                        vectors=[{"id": vector_id, "values": vec, "metadata": metadata}],
                        namespace=namespace,
                    )
                    try:
                        sb.table("memories").update({"embedding_id": vector_id}).eq("id", memory_id).execute()
                    except Exception:
                        pass

                updated.append({"idx": idx, "memory_id": memory_id, "hd": best_hd})
                continue
            except Exception:
                # fall through to insert-as-new
                pass

        # ---- insert brand-new row
        payload = {
            "type": mem_type,
            "title": title,
            text_col: text,
            "summary": summary,
            "tags": tagset,
            "source": source,
            "role_view": role_view,
            "file_id": file_id,
            "dedupe_hash": dedupe_hash,
            "simhash64": sh_s,  # signed
        }
        sb.table("memories").insert(payload).execute()

        # fetch id
        sel = sb.table("memories").select("id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        data = sel.data if hasattr(sel, "data") else sel.get("data")
        if not data:
            skipped.append({"idx": idx, "reason": "insert_select_missed"})
            continue
        memory_id = data[0]["id"]

        # embed + upsert vector
        vec = embed(text)
        if vec:
            try:
                vector_id = f"mem_{memory_id}"
                namespace = {"semantic": "semantic", "episodic": "episodic", "procedural": "procedural"}[mem_type]
                eid_list = link_entities(sb, memory_id, llm_entities(text))
                metadata = {
                    "type": mem_type,
                    "title": title,
                    "tags": tagset,
                    "created_at": now_iso(),
                    "role_view": role_view,
                    "entity_ids": eid_list,
                    "source": source,
                }
                pinecone_index.upsert(
                    vectors=[{"id": vector_id, "values": vec, "metadata": metadata}],
                    namespace=namespace,
                )
                try:
                    sb.table("memories").update({"embedding_id": vector_id}).eq("id", memory_id).execute()
                except Exception:
                    pass
            except Exception:
                pass
        else:
            # still extract+link entities for graph even if embedding failed
            try:
                link_entities(sb, memory_id, llm_entities(text))
            except Exception:
                pass

        created.append({"idx": idx, "memory_id": memory_id})

    # final, correctly indented return
    return {"upserted": created, "skipped": skipped, "updated": updated}
