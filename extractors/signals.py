# extractors/signals.py
import os, json
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """Extract high-signal facts from a document.
Return STRICT JSON:

{
 "decisions":[{"title":"","text":"","entities":[],"confidence":0.0}],
 "deadlines":[{"title":"","text":"","date":"","owner":"","entities":[],"confidence":0.0}],
 "procedures":[{"title":"","text":"","steps":[],"entities":[],"confidence":0.0}],
 "entities":[{"title":"","text":"","entity_type":"","aliases":[],"confidence":0.0}]
}

Guidelines:
- decisions: approvals, rejections, commitments, changes in course
- deadlines: due dates (include date and owner if available)
- procedures: new/updated SOPs or rules (summarize steps)
- entities: projects, teams, partners, key people, artifacts
If none, return empty arrays. Never include prose outside JSON.
"""

def _safe_load_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"decisions": [], "deadlines": [], "procedures": [], "entities": []}

def extract_signals_from_text(title: str, text: str) -> Dict[str, List[Dict[str, Any]]]:
    if not text or not text.strip():
        return {"candidates": [], "raw": {"decisions": [], "deadlines": [], "procedures": [], "entities": []}}

    model = os.getenv("EXTRACTOR_MODEL", "gpt-4.1-mini")
    payload = f"Title: {title}\n\nText:\n{text[:200000]}"  # safety cap

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":payload}],
        temperature=0
    )
    raw = resp.choices[0].message.content or "{}"
    data = _safe_load_json(raw)

    candidates: List[Dict[str, Any]] = []

    for d in data.get("decisions", []):
        candidates.append({
            "fact_type": "decision",
            "title": d.get("title") or "Decision",
            "text": d.get("text") or "",
            "tags": ["source:upload","type:decision", f"title:{title}"],
            "confidence": float(d.get("confidence") or 0.0),
        })

    for d in data.get("deadlines", []):
        date = (d.get("date") or "").strip()
        owner = (d.get("owner") or "").strip()
        line = d.get("text") or ""
        if date: line += f" | date: {date}"
        if owner: line += f" | owner: {owner}"
        candidates.append({
            "fact_type": "deadline",
            "title": d.get("title") or "Deadline",
            "text": line,
            "tags": ["source:upload","type:deadline", f"title:{title}"],
            "confidence": float(d.get("confidence") or 0.0),
        })

    for p in data.get("procedures", []):
        steps = p.get("steps") or []
        steps_text = "\n".join(f"- {s}" for s in steps)
        body = (p.get("text") or "").strip()
        if steps_text:
            body = (body + "\n\nSteps:\n" + steps_text).strip()
        candidates.append({
            "fact_type": "procedure",
            "title": p.get("title") or "Procedure",
            "text": body,
            "tags": ["source:upload","type:procedure", f"title:{title}"],
            "confidence": float(p.get("confidence") or 0.0),
        })

    for e in data.get("entities", []):
        summary = e.get("text") or e.get("title") or ""
        candidates.append({
            "fact_type": "entity",
            "title": e.get("title") or "Entity",
            "text": summary,
            "tags": ["source:upload","type:entity", f"title:{title}"],
            "confidence": float(e.get("confidence") or 0.0),
        })

    return {"candidates": candidates, "raw": data}
