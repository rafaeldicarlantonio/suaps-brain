# vendors/supabase_client.py
import os
from typing import Optional
from supabase import create_client, Client

_client: Optional[Client] = None

def get_client() -> Client:
    """Return a cached Supabase Client using SERVICE ROLE credentials."""
    global _client
    if _client is not None:
        return _client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    _client = create_client(url, key)
    return _client

# ---- Backward-compat shim ----
# Some legacy code does: `from vendors.supabase_client import supabase` then `supabase.table("...")`.
# Provide an object with .table/.schema/.rpc/.storage that forwards to the real client.

class _Compat:
    def table(self, name: str):
        return get_client().table(name)
    def schema(self, name: str):
        return get_client().schema(name)
    @property
    def rpc(self):
        return get_client().rpc
    @property
    def storage(self):
        return get_client().storage

# Exported symbol used by legacy code
supabase = _Compat()

# Optional convenience for new code
def table(name: str):
    return get_client().table(name)
