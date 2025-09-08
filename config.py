# config.py
import os

REQUIRED = [
    "OPENAI_API_KEY","PINECONE_API_KEY","PINECONE_ENV","PINECONE_INDEX",
    "SUPABASE_URL","SUPABASE_SERVICE_ROLE_KEY","X_API_KEY",
    "CHAT_MODEL","REVIEWER_MODEL","EXTRACTOR_MODEL","EMBED_MODEL",
]

missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing env vars: {missing}")

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
TOPK_PER_TYPE = int(os.getenv("TOPK_PER_TYPE", "30"))
RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))
RECENCY_FLOOR = float(os.getenv("RECENCY_FLOOR", "0.35"))
