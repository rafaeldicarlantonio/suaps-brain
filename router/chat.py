"""
router/chat.py
--------------
Chat endpoint with role input and simple intent routing per PRD.
Create this file only if you don't already have a /chat route elsewhere.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from agent import pipeline as _pipeline
except Exception:
    _pipeline = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

class ChatInput(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    role: Optional[str] = Field(None, description="researcher|staff|director|admin")
    session_id: Optional[str] = None
    message: str
    history: List[Dict[str, Any]] = []
    temperature: Optional[float] = None
    preferences: Optional[Dict[str, Any]] = None
    debug: Optional[bool] = False

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Dict[str, Any]]
    guidance_questions: List[str]
    autosave: Dict[str, Any]
    redteam: Dict[str, Any]
    metrics: Dict[str, Any]

def classify_intent(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["upload", "attach", "ingest"]):
        return "ingest"
    if any(k in t for k in ["debug", "health", "status"]):
        return "admin"
    return "qa"

@router.post("", response_model=ChatResponse)
def chat_endpoint(body: ChatInput, x_api_key: Optional[str] = Header(None)) -> Any:
    if _pipeline is None or not hasattr(_pipeline, "chat"):
        raise HTTPException(500, "pipeline.chat unavailable")

    intent = classify_intent(body.message)
    # For MVP, we still call pipeline.chat; branch later as needed.

    sid, resp = _pipeline.chat(
        user_id=body.user_id,
        user_email=body.user_email,
        role=body.role,
        session_id=body.session_id,
        message=body.message,
        history=body.history,
        temperature=body.temperature,
        debug=bool(body.debug),
    )
    return resp
