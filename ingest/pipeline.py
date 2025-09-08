# ingest/pipeline.py
import os, re, math, hashlib, datetime
from typing import List, Dict, Any, Optional, Tuple

def now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(s: str, chunk_size: int, overlap: int) -> List[str]:
    s = s.strip()
    if not s: return []
    chunks, i, n = [], 0, len(s)
    step = max(1, chunk_size - overlap)
    while i < n and len(chunks) < int(os.getenv("MAX_CHUNKS_PER_FILE", "500")):
        chunk = s[i:i+chunk_size]
        chunks.append(chunk)
        i += step
    return chunks

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def upsert_memories_from_chunks(*, sb, pinecone_index, embedder, file_id: Optional[str], title_prefix: str, chunks: List[str], mem_type: str = "semantic", tags: Optional[List[str]] = None, role_view: Optional[List[str]] = None, source: str = "upload", text_col_env: str = "value") -> Dict[str, Any]:
    tags = tags or []
    role_view = role_view or []
    text_col = (os.getenv("MEMORIES_TEXT_COLUMN") or text_col_env).strip().lower()
    created, skipped = [], []

    # embed function (vector length depends on model)
    def embed(text: str) -> Optional[List[float]]:
        try:
            from openai import OpenAI
            oai = OpenAI()
            kwargs = {"model": os.getenv("EMBED_MODEL","text-embedding-3-small"), "input": text}
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
        dedupe_hash = sha256(text)

        # duplicate check
        existing = sb.table("memories").select("id,embedding_id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        rows = existing.data if hasattr(existing, "data") else existing.get("data")
        if rows:
            skipped.append({"idx": idx, "reason": "duplicate", "memory_id": rows[0]["id"]})
            continue

        # insert into memories
        title = f"{title_prefix} — part {idx+1}"
        payload = {
            "type": mem_type,
            "title": title,
            text_col: text,
            "tags": tags,
            "source": source,
            "role_view": role_view,
            "file_id": file_id,
            "dedupe_hash": dedupe_hash,
        }
        ins = sb.table("memories").insert(payload).execute()

        # fetch id
        sel = sb.table("memories").select("id").eq("dedupe_hash", dedupe_hash).limit(1).execute()
        data = sel.data if hasattr(sel, "data") else sel.get("data")
        if not data:
            skipped.append({"idx": idx, "reason": "insert_select_missed"})
            continue
        memory_id = data[0]["id"]

        # embed + pinecone
        vec = embed(text)
        if vec:
            try:
                vector_id = f"mem_{memory_id}"
                namespace = {"semantic":"semantic","episodic":"episodic","procedural":"procedural"}[mem_type]
                metadata = {
                    "type": mem_type,
                    "title": title,
                    "tags": tags,
                    "created_at": now_iso(),
                    "role_view": role_view,
                    "entity_ids": [],
                    "source": source,
                    "reason": "ingest",
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

        created.append({"idx": idx, "memory_id": memory_id})

    return {"upserted": created, "skipped": skipped}
