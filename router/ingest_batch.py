from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, Body

from agent.ingest import ingest_batch as do_ingest  # expects (items: List[dict], dedupe: bool)

router = APIRouter(tags=["ingest"])

def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

def _normalize_payload(payload: Any) -> (List[Dict[str, Any]], bool):
    """
    Accept either:
    - {"items": [ {title,text,type,...}, ... ], "dedupe": true}
    - [ {title,text,type,...}, ... ]  (top-level array)
    Also tolerate alternate field names: "documents"/"docs".
    """
    if isinstance(payload, list):
        items = payload
        dedupe = True
    elif isinstance(payload, dict):
        items = payload.get("items") or payload.get("documents") or payload.get("docs")
        dedupe = bool(payload.get("dedupe", True))
    else:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if items is None:
        raise HTTPException(status_code=400, detail="Missing 'items' array")

    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="'items' must be an array")

    # minimal normalization of each item
    norm: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            raise HTTPException(status_code=400, detail="Each item must be an object")
        title = it.get("title") or it.get("name") or "Untitled"
        text = it.get("text") or it.get("content")
        mtype = it.get("type") or "semantic"
        if not text:
            # skip empty text; server should not blow up
            continue
        norm.append({
            "title": title,
            "text": text,
            "type": mtype,
            "tags": it.get("tags") or [],
            "source": it.get("source") or "ingest",
            "role_view": it.get("role_view"),
            "file_id": it.get("file_id"),
        })
    if not norm:
        raise HTTPException(status_code=400, detail="No valid items to ingest")
    return norm, dedupe

@router.post("/ingest/batch")
def ingest_batch(
    payload: Any = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)
    try:
        items, dedupe = _normalize_payload(payload)
        res = do_ingest(items, dedupe=dedupe)  # returns {"upserted":[...], "skipped":[...]}
        # shape-guard the response
        up = [{"memory_id": u.get("memory_id"), "embedding_id": u.get("embedding_id")} for u in (res.get("upserted") or [])]
        sk = [{"reason": (s.get("reason") or "unknown")} for s in (res.get("skipped") or [])]
        return {"upserted": up, "skipped": sk}
    except HTTPException:
        raise
    except Exception as ex:
        # keep message concise; don’t expose internals
        raise HTTPException(status_code=500, detail=f"ingest failed")
