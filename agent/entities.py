# agent/entities.py
# Minimal entity extraction + upsert (PRD §6.6)

from __future__ import annotations
from typing import List, Tuple, Dict, Any

from vendors.openai_client import client, CHAT_MODEL
from agent import store

SYSTEM = (
    "Extract entities from the provided text. "
    "Return a JSON object with a single key 'entities' as a list of items, "
    "each item = {name: string, type: one of ['person','org','project','artifact','concept']}."
)

def extract_entities(text: str) -> List[Tuple[str, str]]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text[:8000]},
    ]
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=msgs,
        )
        js = resp.choices[0].message
        import json
        data = json.loads(js.content)
        out = []
        for e in data.get("entities", []):
            name = (e.get("name") or "").strip()
            typ = (e.get("type") or "").strip()
            if name and typ in ("person","org","project","artifact","concept"):
                out.append((name, typ))
        return out
    except Exception:
        return []

def upsert_entities_for_memory(memory_id: str, text: str) -> Dict[str, Any]:
    ents = extract_entities(text)
    eids = []
    for name, typ in ents:
        eid = store.ensure_entity(name, typ)
        store.insert_entity_mention(eid, memory_id, 1.0)
        eids.append(eid)
    return {"entity_ids": eids, "count": len(eids)}
