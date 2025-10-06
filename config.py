# config.py
import os

REQUIRED = [
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_ANON_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
]

# v3 client doesn’t use PINECONE_ENV — don’t require it
# OPTIONALS with sensible defaults
DEFAULTS = {
    "EMBED_MODEL": "text-embedding-3-small",
    "MEMORIES_TEXT_COLUMN": "text",
    "EMBED_DIM": None,     # None = let model decide / match index
    "X_API_KEY": None,     # optional external auth for your endpoints
}

def load_config():
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")
    cfg = {k: os.getenv(k) for k in REQUIRED}
    for k, v in DEFAULTS.items():
        val = os.getenv(k, v)
        cfg[k] = int(val) if (k == "EMBED_DIM" and val) else val
    return cfg

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
TOPK_PER_TYPE = int(os.getenv("TOPK_PER_TYPE", "30"))
RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))
RECENCY_FLOOR = float(os.getenv("RECENCY_FLOOR", "0.35"))
