from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, Body

# call the self-contained ingestion
from agent.ingest import ingest_batch as do_ingest

router = APIRouter()

def _auth(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH", "false").lower() == "true":
        return
    if not want or x_api_key != want:
        raise HTTPException(status_code=401, detail="invalid X-API-Key")

@router.post("/ingest/batch")  # final path EXACTLY /ingest/batch
def ingest_batch_handler(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _auth(x_api_key)

    items: List[Dict[str, Any]] = payload.get("items") or []
    dedupe: bool = bool(payload.get("dedupe", True))

    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="items must be a non-empty array")

    try:
        return do_ingest(items=items, dedupe=dedupe)
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"ingest_failed: {ex}")
