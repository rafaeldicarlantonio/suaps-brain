from typing import List, Dict
from agent import retrieval

def build_working_window(messages: List[Dict], max_turns: int = 10):
    # naive: last N messages; can be upgraded to token-based window
    return messages[-max_turns:]

def fetch_context(user_id: str, user_query: str):
    matches = retrieval.search(user_id=user_id, query=user_query, top_k=6, types=["episodic","semantic"])
    ctx = []
    for m in matches:
        md = m.metadata or {}
        snippet = md.get("title","")
        ctx.append(f"[{md.get('type','?')}] {snippet} — {md.get('created_at','')}")
    return ctx[:6]
