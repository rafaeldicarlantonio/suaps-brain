# vendors/pinecone_client.py
import os
from pinecone import Pinecone

_pc = None
_index = None

def get_index():
    """
    Returns a cached Pinecone Index object using PINECONE_API_KEY and PINECONE_INDEX.
    """
    global _pc, _index
    if _index is not None:
        return _index

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX")
    if not api_key or not index_name:
        raise RuntimeError("Missing PINECONE_API_KEY or PINECONE_INDEX")

    _pc = Pinecone(api_key=api_key)
    _index = _pc.Index(index_name)
    return _index
