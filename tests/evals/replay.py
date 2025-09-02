import json
import os
import sys
import time
from typing import List, Dict, Any

import requests

BASE = os.getenv("EVAL_BASE_URL", "http://localhost:10000")
API_KEY = os.getenv("ACTIONS_API_KEY", "dev_key")

def fail(msg: str):
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)

def post_chat(prompt: str, role: str) -> Dict[str, Any]:
    t0 = time.time()
    r = requests.post(
        f"{BASE}/chat",
        headers={"X-API-Key": API_KEY},
        json={"message": prompt, "role": role, "debug": False},
        timeout=30,
    )
    dt = int((time.time() - t0) * 1000)
    if r.status_code != 200:
        fail(f"/chat HTTP {r.status_code}: {r.text}")
    data = r.json()
    data["_latency_ms"] = dt
    return data

def run_case(case: Dict[str, Any]) -> None:
    prompt = case["prompt"]
    role = case.get("role", "researcher")
    res = post_chat(prompt, role)
    ans = (res.get("answer") or "").lower()
    if not ans:
        fail("empty answer")

    # Must-include checks
    for needle in case.get("must_include", []):
        if needle.lower() not in ans:
            fail(f"missing must_include: {needle}")

    # Citation tag check (allow any-match across provided tags)
    tags_any: List[str] = case.get("must_cite_tags_any", [])
    if tags_any:
        cits = res.get("citations") or []
        matched = False
        for c in cits:
            tags = (c.get("tags") if isinstance(c, dict) else []) or []
            if any(t in tags for t in tags_any):
                matched = True
                break
        if not matched:
            fail(f"citations missing required tags_any: {tags_any}")

    # Latency check
    max_ms = int(case.get("max_latency_ms", 3000))
    if res["_latency_ms"] > max_ms:
        fail(f"latency {res['_latency_ms']}ms > {max_ms}ms")

    # Red-team block check
    rt = res.get("redteam") or {}
    if isinstance(rt, dict) and rt.get("action") == "block":
        fail("red-team returned block on known-good case")

    print(f"[OK] {case['id']} in {res['_latency_ms']}ms")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/evals/golden_set.json"
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for case in cases:
        run_case(case)

if __name__ == "__main__":
    main()
