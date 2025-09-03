from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import APIRouter, Body, Header, HTTPException

router = APIRouter(tags=["chat"])

@router.post("/chat")
def chat_endpoint(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    # TODO: plug in your real chat handler here. For now we echo.
    prompt = (payload or {}).get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    return {"answer": f"ECHO: {prompt}", "citations": [], "guidance_questions": []}
