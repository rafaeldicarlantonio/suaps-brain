import os
from openai import OpenAI

# Single client instance. Requires OPENAI_API_KEY in the environment.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model env names:
# Prefer PRD vars (CHAT_MODEL, EMBED_MODEL); fall back to legacy ones.
CHAT_MODEL  = os.getenv("CHAT_MODEL")  or os.getenv("OPENAI_CHAT_MODEL")  or "gpt-4.1-mini"
EMBED_MODEL = os.getenv("EMBED_MODEL") or os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small"
