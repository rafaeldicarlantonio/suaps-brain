from __future__ import annotations

import os
from typing import Optional, Dict, Any
from fastapi import APIRouter, Body, Header, HTTPException
from pydantic import BaseModel

from agent.pipeline import handle_chat

router = APIRouter(tags=["chat"])

# Simple API key guard (PRD §15)
def _require_key(x_api_key: Optional[str]):
    want = os.getenv("X_API_KEY") or os.getenv("ACTIONS_API_KEY")
    if os.getenv("DISABLE_AUTH","false").lower() == "true":
        return
    if not want:
        raise HTTPException(status_code=500, detail="Server missing X_API_KEY")
    if not x_api_key or x_api_key != want:
        raise HTTPException(status_code=401, detail="unauthorized")

class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    role: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    debug: Optional[bool] = False

@router.post("/chat")
def chat_endpoint(
    payload: ChatRequest = Body(...),
    x_api_key: Optional[str] = Header(None),
):
    _require_key(x_api_key)
    if not payload.prompt or not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        return handle_chat(payload.dict())
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"/chat error: {ex}")
