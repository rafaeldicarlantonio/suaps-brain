from __future__ import annotations

import os
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Header, HTTPException, Body

# Use the single-item ingest used by /upload (this path is known-good)
from agent.ingest import ingest_text

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
    - {"items": [ {...}, ... ], "dedupe": true}
    - [ {...}, ... ]
    Also tolerate alternate field names: "documents"/"docs".
    Each item must contain at least a 'text' (or 'content').
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

    norm: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            raise HTTPException(status_code=400, detail="Each item must be an object")

        text = (it.get("text") or it.get("content") or "").strip()
        if not text:
            # Skip empty text; do not fail the whole batch
            continue

        norm.append({
            "title": (it.get("title") or it.get("name") or "Untitled").strip(),
            "text": text,
            "type": (it.get("type") or "semantic"),
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

        upserted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []

        for it in items:
            try:
                # Call the known-good single-item path
                res = ingest_text(
                    text=it["text"],
                    title=it["title"],
                    type=it["type"],
                    tags=it["tags"],
                    source=it["source"],
                    role_view=it["role_view"],
                    file_id=it["file_id"],
                    # If your ingest_text supports dedupe internally, it will honor it.
                    # If not, it's still safe; Day 2 will add explicit near-dup checks.
                )
                # Normalize response shape
                emb_id = None
                mem_id = None
                if isinstance(res, dict):
                    up = (res.get("upserted") or [])
                    if up and isinstance(up, list):
                        emb_id = (up[0] or {}).get("embedding_id")
                        mem_id = (up[0] or {}).get("memory_id")
                    else:
                        # Fallbacks if ingest_text returns a direct dict
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
