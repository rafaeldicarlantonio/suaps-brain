from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, Body, Request

from agent.ingest import ingest_batch as do_ingest  # your existing ingest

router = APIRouter()

# --- auth helpers (accept multiple formats to avoid Action quirks) ---
def _want_key() -> Optional[str]:
    return os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")

def _auth(request: Request,
          x_api_key: Optional[str],
          authorization: Optional[str],
          payload: Optional[Dict[str, Any]]):
    if os.getenv("DISABLE_AUTH", "false").lower() == "true":
        return
    want = _want_key()
    if not want:
        raise HTTPException(status_code=401, detail="server missing API key config")

    supplied = None
    if x_api_key:
        supplied = x_api_key
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization.split(" ", 1)[1].strip()
    if not supplied:
        supplied = request.query_params.get("x_api_key")
    if not supplied and isinstance(payload, dict):
        supplied = payload.get("x_api_key") or payload.get("api_key")

    if supplied != want:
        raise HTTPException(status_code=401, detail="invalid X-API-Key")

# --- endpoint ---
@router.post("/ingest/batch")
def ingest_batch_handler(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    _auth(request, x_api_key, authorization, payload)

    items: List[Dict[str, Any]] = payload.get("items") or []
    dedupe: bool = bool(payload.get("dedupe", True))
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="items must be a non-empty array")

    try:
        res = do_ingest(items=items, dedupe=dedupe)
        up = len(res.get("upserted", []))
        sk = len(res.get("skipped", []))
        summary = {"upserted_count": up, "skipped_count": sk}
        res["summary"] = summary

        # <<< Patch 2A.2: if nothing changed, fail with 422 so GPT must show the error >>>
        if up == 0:
            raise HTTPException(status_code=422, detail={"message": "No items ingested", "summary": summary})

        return res
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"ingest_failed: {ex}")
