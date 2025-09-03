from __future__ import annotations

import os
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Body

from agent.ingest import ingest_batch as do_ingest
from schemas.api import IngestBatchRequest, IngestBatchResponse

router = APIRouter(tags=["ingest"])

def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

@router.post("/ingest/batch", response_model=IngestBatchResponse)
def ingest_batch(
    payload: IngestBatchRequest = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)
    try:
        res = do_ingest([i.model_dump() for i in payload.items], dedupe=bool(payload.dedupe))
        # coerce to response model shape
        return IngestBatchResponse(
            upserted=[{"memory_id": u.get("memory_id"), "embedding_id": u.get("embedding_id")} for u in res.get("upserted", [])],
            skipped=[{"reason": s.get("reason", "unknown")} for s in res.get("skipped", [])],
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"ingest failed: {ex}")
