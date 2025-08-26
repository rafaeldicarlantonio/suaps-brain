from vendors.openai_client import client, CHAT_MODEL
from agent.prompts import build_system
from agent.memory_router import build_working_window, fetch_context
from agent import retrieval, store
import json

def propose_memory(user_msg: str, assistant_msg: str):
    prompt = f"""Given the user message and assistant reply, decide if a durable memory should be saved.
Return JSON with fields: save(bool), type(one of episodic|semantic), title, summary, tags(list), importance(int 1-5).
User: {user_msg}
Assistant: {assistant_msg}
"""
    out = client.chat.completions.create(model=CHAT_MODEL, messages=[
        {"role":"system","content":"Return strict JSON only."},
        {"role":"user","content": prompt}
    ], temperature=0.2)
    try:
        return json.loads(out.choices[0].message.content)
    except Exception:
        return {"save": False}

def run_chat(user_id: str, session_id: str, messages: list, user_msg: str):
    # Working memory
    working = build_working_window(messages)
    # Context retrieval
    ctx = fetch_context(user_id, user_msg)

    system = build_system()
    convo = [{"role":"system","content":system}]
    for m in working:
        convo.append({"role": m["role"], "content": m["content"]})
    if ctx:
        convo.append({"role":"system","content":"Relevant context (summaries):\n" + "\n".join(ctx)})
    convo.append({"role":"user","content":user_msg})

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=convo)
    answer = resp.choices[0].message.content

    # write back candidate memory
    proposal = propose_memory(user_msg, answer)
    if proposal.get("save") and int(proposal.get("importance",3)) >= 4:
        row = store.upsert_memory(
            user_id=user_id,
            type_=proposal.get("type","episodic"),
            title=proposal.get("title",""),
            content=proposal.get("summary", answer[:1000]),
            importance=int(proposal.get("importance",4)),
            tags=proposal.get("tags",[])
        )
        retrieval.upsert_memory_vector(
            mem_id=row["id"], user_id=user_id, type_=row["type"],
            content=row["content"], title=row.get("title",""),
            tags=row.get("tags",[]), importance=row.get("importance",3)
        )
    return answer
