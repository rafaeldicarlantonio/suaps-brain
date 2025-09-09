# guardrails/redteam.py
import os, json
from typing import Dict, Any, List
from openai import OpenAI

def review_answer(*, draft_json: Dict[str, Any], prompt: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call the reviewer model. retrieved_chunks: list of {"id": str, "text": str}
    """
    client = OpenAI()
    sys = _load("prompts/system_reviewer.md", fallback="You are a strict reviewer. Return JSON.")
    user = json.dumps({
        "prompt": prompt,
        "draft": draft_json,
        "retrieved_chunks": retrieved_chunks[:12],  # cap to keep prompt size sane
    })
    r = client.chat.completions.create(
        model=os.getenv("REVIEWER_MODEL", os.getenv("CHAT_MODEL","gpt-4.1-mini")),
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        temperature=0,
    )
    try:
        return json.loads(r.choices[0].message.content or "{}")
    except Exception:
        return {"action":"allow","reasons":["parse_error"]}
    
def _load(path: str, fallback: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback
