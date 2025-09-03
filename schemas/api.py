from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ---- Chat ----
class AutosaveItem(BaseModel):
    memory_id: str
    type: str
    title: Optional[str] = None

class AutosaveResult(BaseModel):
    saved: bool
    items: List[AutosaveItem] = []

class RedteamResult(BaseModel):
    action: str
    reasons: List[str] = []

class ChatRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    role: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    debug: Optional[bool] = False

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[str]
    guidance_questions: List[str]
    autosave: AutosaveResult
    redteam: RedteamResult
    metrics: Dict[str, Any]

# ---- Upload ----
class UploadResponse(BaseModel):
    file_id: str
    bytes: int
    mime_type: str
    ingest_job_id: Optional[str] = None

# ---- Ingest ----
class IngestItem(BaseModel):
    title: str
    text: str
    type: str
    tags: List[str] = []
    source: str
    role_view: Optional[List[str]] = None
    file_id: Optional[str] = None

class IngestBatchRequest(BaseModel):
    items: List[IngestItem]
    dedupe: Optional[bool] = True

class IngestUpserted(BaseModel):
    memory_id: Optional[str] = None
    embedding_id: Optional[str] = None

class IngestSkipped(BaseModel):
    reason: str

class IngestBatchResponse(BaseModel):
    upserted: List[IngestUpserted] = []
    skipped: List[IngestSkipped] = []

# ---- Memories / upsert ----
class MemoriesUpsertRequest(BaseModel):
    type: str
    title: Optional[str] = None
    text: str
    tags: List[str] = []
    role_view: Optional[List[str]] = None

class MemoriesUpsertResponse(BaseModel):
    memory_id: str
    embedding_id: Optional[str] = None

# ---- Debug / Health ----
class DebugMemRow(BaseModel):
    id: str
    type: Optional[str] = None
    title: Optional[str] = None
    created_at: Optional[str] = None

class DebugMemoriesResponse(BaseModel):
    items: List[DebugMemRow]

class DebugSelftestResponse(BaseModel):
    openai_ms: int
    pinecone_ok: bool
    latency_ms: int

class HealthzResponse(BaseModel):
    status: str = "ok"
    openai: Dict[str, Any]
    pinecone: Dict[str, Any]
    supabase: Dict[str, Any]
