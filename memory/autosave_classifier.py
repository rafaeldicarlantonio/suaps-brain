# memory/autosave_classifier.py
"""
Lightweight importance classifier for autosave candidates.
Uses the EXTRACTOR_MODEL to decide if a fact is critical enough to store.
"""

import os
import json
from typing import Dict, Any
from openai import OpenAI

client = OpenAI()

IMPORTANCE_PROMPT = """You are an assistant that classifies organizational facts by importance.
Levels:
- high: critical decisions, deadlines, policies, procedures, approvals
- medium: important but not binding (suggestions, ideas, discussions)
- low: minor chatter or non-actionable notes

Return ONLY JSON:
{
  "importance": "high" | "medium" | "low",
  "importance_score": float  // 0.0–1.0, confidence that this fact is important
}

Fact text: {text}
"""

def classify_importance(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs an LLM classification on a candidate fact.
    Returns a dict with keys: importance, importance_score.
    """
    text = candidate.get("text") or ""
    if not text.strip():
        return {"importance": "low", "importance_score": 0.0}

    model = os.getenv("EXTRACTOR_MODEL", "gpt-4.1-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": IMPORTANCE_PROMPT.format(text=text)},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        return {
            "importance": data.get("importance", "low"),
            "importance_score": float(data.get("importance_score", 0.0)),
        }
    except Exception:
        return {"importance": "low", "importance_score": 0.0}
