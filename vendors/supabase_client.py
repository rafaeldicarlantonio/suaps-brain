# vendors/supabase_client.py
import os

# Expose the supabase module itself for legacy imports like:
#   from vendors.supabase_client import supabase
try:
    import supabase as supabase  # module alias (supabase-py package)
except Exception:
    supabase = None  # keep attribute present to avoid ImportError

from supabase import create_client, Client

_client: Client | None = None

def get_client() -> Client:
    """
    Cached Supabase client using the SERVICE ROLE key.
    Raises a clear error if envs are missing.
    """
    global _client
    if _client is not None:
        return _client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    _client = create_client(url, key)
    return _client
