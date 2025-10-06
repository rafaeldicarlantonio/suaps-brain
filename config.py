# config.py — sane config with loud failures

import os

# Hard requirements. Fail fast if any are missing.
REQUIRED = [
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX",
]

# Optional knobs with defaults that won't sandbag you at runtime.
DEFAULTS = {
    "EMBED_MODEL": "text-embedding-3-small",
    "MEMORIES_TEXT_COLUMN": "text",
    "EMBED_DIM": None,   # None = let model decide / must match index
    "X_API_KEY": None,   # optional request auth for your endpoints
}

def load_config():
    """
    Load env config, erroring clearly if anything critical is missing.
    Returns a dict of required + defaults (with types normalized).
    """
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")

    cfg = {k: os.getenv(k) for k in REQUIRED}

    for k, v in DEFAULTS.items():
        val = os.getenv(k, v)
        if k == "EMBED_DIM" and val is not None and val != "":
            try:
                val = int(val)
            except Exception:
                raise RuntimeError("EMBED_DIM must be an integer if provided")
        cfg[k] = val

    return cfg

# Misc operational tuning. Keep these here so you don’t hardcode magic numbers elsewhere.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
TOPK_PER_TYPE = int(os.getenv("TOPK_PER_TYPE", "30"))
RECENCY_HALFLIFE_DAYS = int(os.getenv("RECENCY_HALFLIFE_DAYS", "90"))
RECENCY_FLOOR = float(os.getenv("RECENCY_FLOOR", "0.35"))
