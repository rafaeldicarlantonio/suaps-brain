# SUAPS Agent (FastAPI + Supabase + Pinecone)

Production-ready wrapper around the ideas in `ALucek/agentic-memory`:
- Working, Episodic, Semantic, Procedural memories
- API endpoints for Chat + Memory upsert + Batch ingestion
- OpenAPI spec for ChatGPT **Actions** UI

## 0) Prereqs
- Python 3.11+
- Supabase project (get `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`)
- Pinecone account & index (serverless) matching embedding dim 1536
- OpenAI API key

## 1) Setup
```bash
git clone <this-repo-or-zip>
cd suaps-agent
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## 2) Supabase schema
Paste `supabase_schema.sql` in the Supabase SQL Editor and run.
(Optional) Add RLS later.

## 3) Pinecone index
```bash
python scripts/create_pinecone_index.py
```

## 4) Run the API
```bash
uvicorn app:app --reload
```

Health check: http://localhost:8000/healthz

## 5) Test chat
```bash
curl -X POST http://localhost:8000/chat  -H "Content-Type: application/json" -H "X-API-Key: dev_key"  -d '{"user_id":"usr_demo","message":"Quick test","history":[]}'
```

## 6) Ingest documents
Put files in `./docs` and run:
```bash
INGEST_USER_ID=<uuid> python scripts/ingest_from_files.py ./docs/handbook.pdf ./docs/policy.md
```
- Chunking mirrors the demo style: **character-based with overlap**, configurable via env:
  - `CHUNK_SIZE` (default 4000 chars), `CHUNK_OVERLAP` (default 400)
- Each chunk is distilled to a **semantic** memory (summary + optional Q&A)
- Memories are saved in Supabase and upserted to Pinecone

## 7) GPT Actions
- Upload `openapi.yaml` in your GPT → Actions
- Auth: **API Key** with header `X-API-Key`
- In the GPT system prompt, instruct the GPT to use `/chat` for each message and reuse `session_id`

## Notes
- Retrieval top-k=6 split across episodic/semantic, adjustable in `agent/memory_router.py`
- Write-back gate: importance ≥ 4 (see `agent/pipeline.py`)
- You can add org-level filters later by storing `org_id` in Supabase + Pinecone metadata
