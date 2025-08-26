import os, re, json, hashlib
from typing import List
from vendors.openai_client import client
from agent import store, retrieval

SUMMARIZER_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))
MAKE_QA = os.getenv("MAKE_QA", "true").lower() == "true"
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.92"))

def llm_json(prompt: str, text: str):
    msg = [
        {"role":"system","content":"Return strict JSON only, no commentary."},
        {"role":"user","content": prompt + "\n\n---\n" + text[:8000]}
    ]
    out = client.chat.completions.create(model=SUMMARIZER_MODEL, messages=msg, temperature=0.2)
    import json
    return json.loads(out.choices[0].message.content)

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t.strip()

def chunk_text(t: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t = clean_text(t)
    chunks, i = [], 0
    n = len(t)
    if n == 0:
        return []
    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(t[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

SUMMARIZE_PROMPT = """You are preparing durable knowledge from a document chunk.
Return concise JSON with fields:
- title (<= 90 chars)
- summary (200-400 words)
- tags (3-8 short tokens)
Use neutral, factual language. Do not include PII unless it is already public."""

QA_PROMPT = """Generate 2-5 helpful Q&A pairs that cover key facts from the input text.
Return JSON list of objects: [{"question": "...", "answer": "..."}].
Questions should be specific; answers concise and self-contained."""

def _dedupe_hit(user_id: str, summary: str, tags: list[str]) -> bool:
    # query Pinecone with the distilled summary and require clickup/wiki tags presence
    matches = retrieval.search(user_id=user_id, query=summary, top_k=1, types=["semantic"])
    if not matches:
        return False
    m = matches[0]
    md = m.metadata or {}
    mtags = set(md.get("tags", []))
    # require intersection on clickup/wiki
    if not {"clickup","wiki"} & mtags:
        return False
    return float(m.score) >= DEDUP_THRESHOLD

def distill_chunk(user_id: str, raw_text: str, base_tags: List[str], make_qa: bool = MAKE_QA, created_at: str | None = None):
    meta = llm_json(SUMMARIZE_PROMPT, raw_text)
    title = meta.get("title","Untitled")
    summary = meta.get("summary", raw_text[:1200])
    tags = list(set((meta.get("tags") or []) + (base_tags or [])))

    # dedupe: if near-duplicate exists with clickup/wiki tags, skip
    if _dedupe_hit(user_id, summary, tags):
        return []

    row = store.upsert_memory(
        user_id=user_id,
        type_="semantic",
        title=title,
        content=summary,
        importance=4,
        tags=tags,
        created_at=created_at
    )
    retrieval.upsert_memory_vector(
        mem_id=row["id"], user_id=user_id, type_=row["type"],
        content=row["content"], title=row["title"], tags=row["tags"],
        importance=row["importance"], created_at_iso=created_at
    )
    created = [row["id"]]

    # Optional QA
    if make_qa:
        try:
            qas = llm_json(QA_PROMPT, raw_text)
            for qa in qas[:5]:
                q_title = (qa.get("question","")[:90]).strip() or (title[:80] + " Q&A")
                q_content = f"Q: {qa['question']}\nA: {qa['answer']}"
                qa_row = store.upsert_memory(
                    user_id=user_id, type_="semantic", title=q_title,
                    content=q_content, importance=4, tags=list(set(tags + ["qa"])),
                    created_at=created_at
                )
                retrieval.upsert_memory_vector(
                    mem_id=qa_row["id"], user_id=user_id, type_=qa_row["type"],
                    content=qa_row["content"], title=qa_row["title"], tags=qa_row["tags"],
                    importance=qa_row["importance"], created_at_iso=created_at
                )
                created.append(qa_row["id"])
        except Exception:
            pass
    return created
