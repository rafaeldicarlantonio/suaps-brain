from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, Body

from agent.ingest import ingest_text  # known-good single-item path

router = APIRouter(tags=["ingest"])

def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

def _normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Accept shapes:
      - {"items":[{...}, ...]}
      - [{"title":...,"text":...}, ...]
      - {"title":...,"text":...}
      - "raw text"
    Normalize to a list of {"title","text","type","tags","source","role_view","file_id"}.
    """
    def norm_item(it: Dict[str, Any]) -> Dict[str, Any]:
        text = (it.get("text") or it.get("content") or "").strip()
        if not text:
            return {}
        return {
            "title": (it.get("title") or it.get("name") or "Untitled").strip(),
            "text": text,
            "type": it.get("type") or "semantic",
            "tags": it.get("tags") or [],
            "source": it.get("source") or "ingest",
            "role_view": it.get("role_view"),
            "file_id": it.get("file_id"),
        }

    items: List[Dict[str, Any]] = []
    if isinstance(payload, str):
        items = [ {"title":"Untitled", "text": payload, "type":"semantic", "tags":[], "source":"ingest"} ]
    elif isinstance(payload, list):
        items = [norm_item(x) for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        raw_items = payload.get("items") or payload.get("documents") or payload.get("docs")
        if isinstance(raw_items, list):
            items = [norm_item(x) for x in raw_items if isinstance(x, dict)]
        else:
            # treat the dict itself as a single item
            items = [norm_item(payload)]
    else:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # prune empties and ensure at least one item with text
    items = [x for x in items if x.get("text")]
    if not items:
        raise HTTPException(status_code=400, detail="No valid items to ingest")
    return items

@router.post("/ingest/batch")
def ingest_batch(
    payload: Any = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)
    try:
        items = _normalize_payload(payload)
        upserted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        for it in items:
            try:
                res = ingest_text(
                    text=it["text"],
                    title=it["title"],
                    type=it["type"],
                    tags=it["tags"],
                    source=it["source"],
                    role_view=it.get("role_view"),
                    file_id=it.get("file_id"),
                )
                emb_id, mem_id = None, None
                if isinstance(res, dict):
                    up = (res.get("upserted") or [])
                    if up and isinstance(up, list):
                        emb_id = (up[0] or {}).get("embedding_id")
                        mem_id = (up[0] or {}).get("memory_id")
                    else:
                        emb_id = res.get("embedding_id")
                        mem_id = res.get("memory_id")
                upserted.append({"memory_id": mem_id, "embedding_id": emb_id})
            except Exception:
                skipped.append({"reason": "ingest_failed"})

        return {"upserted": upserted, "skipped": skipped}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="ingest failed")
