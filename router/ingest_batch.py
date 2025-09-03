from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, Body

from agent.ingest import ingest_batch as do_ingest

router = APIRouter(tags=["ingest"])

def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

@router.post("/ingest/batch")
def ingest_batch(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)
    if not payload or "items" not in payload:
        raise HTTPException(status_code=400, detail="items required")
    try:
        res = do_ingest(payload.get("items", []), dedupe=bool(payload.get("dedupe", True)))
        return res
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"ingest failed: {ex}")
