from __future__ import annotations

import os
from pinecone import Pinecone

# One global client; requires PINECONE_API_KEY in env
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX", "uap-kb")

def get_index():
    """Return a handle to the configured index."""
    return pc.Index(INDEX_NAME)
