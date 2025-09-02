# validators/json.py
# Strict JSON parse + minimal schema check (PRD §7.7)

from __future__ import annotations
from typing import Any, Dict, List
import json

REQUIRED_KEYS = {"answer": str, "citations": list, "guidance_questions": list, "autosave_candidates": list}

def strict_parse_or_retry(raw_content: str, schema: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse JSON; enforce required keys and basic types. Raise ValueError on failure."""
    obj = json.loads(raw_content)
    for k, typ in REQUIRED_KEYS.items():
        if k not in obj:
            raise ValueError(f"missing key: {k}")
        if not isinstance(obj[k], typ):
            raise ValueError(f"wrong type for {k}: expected {typ.__name__}")
    return obj
