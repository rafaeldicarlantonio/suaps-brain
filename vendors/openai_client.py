import os
from openai import OpenAI
--- a/vendors/openai_client.py
+++ b/vendors/openai_client.py
@@ -1,6 +1,11 @@
 import os
 from openai import OpenAI
 
- client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
- EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
- CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-5")
+ # Single client instance; API key from OPENAI_API_KEY (required).
+ client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
+
+ # Model env names:
+ # - Prefer PRD variables (CHAT_MODEL, EMBED_MODEL)
+ # - Fallback to legacy OPENAI_* envs to remain backwards-compatible.
+ CHAT_MODEL  = os.getenv("CHAT_MODEL")  or os.getenv("OPENAI_CHAT_MODEL")  or "gpt-5"
+ EMBED_MODEL = os.getenv("EMBED_MODEL") or os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small"

