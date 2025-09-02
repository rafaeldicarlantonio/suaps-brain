"""
tests/evals/replay.py
---------------------
Runs the golden set against local /chat endpoint and scores:
- Must-include keyphrases present
- Citations include expected tags (any)
- Latency under threshold
- Answer JSON schema basic shape (keys exist)
Usage:
  export BASE_URL=http://localhost:10000
  export X_API_KEY=dev_key
  python -m tests.evals.replay --gold tests/evals/golden_set.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List

import requests

REQUIRED_KEYS = ["answer","citations","guidance_questions","autosave"]

def load_golden(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_chat(base_url: str, api_key: str, prompt: str, role: str = "researcher") -> Dict[str, Any]:
    t0 = time.time()
    r = requests.post(f"{base_url}/chat", headers={
        "Content-Type":"application/json",
        "X-API-Key": api_key
    }, json={
        "message": prompt,
        "role": role
    }, timeout=60)
    latency_ms = int((time.time()-t0)*1000)
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Non-JSON response ({r.status_code}): {r.text[:300]}")
    data.setdefault("metrics",{}).setdefault("latency_ms", latency_ms)
    return data

def has_any_citation_tag(citations: List[Dict[str, Any]], expected_tags_any: List[str]) -> bool:
    exp = set([t.lower() for t in expected_tags_any or []])
    for c in citations or []:
        for t in (c.get("tags") or []):
            if t and t.lower() in exp:
                return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Path to golden_set.json")
    ap.add_argument("--base", default=os.getenv("BASE_URL","http://localhost:10000"))
    ap.add_argument("--key", default=os.getenv("X_API_KEY","dev_key"))
    args = ap.parse_args()

    cases = load_golden(args.gold)
    ok = 0
    fail = 0
    details: List[str] = []

    for case in cases:
        cid = case.get("id","<no-id>")
        prompt = case["prompt"]
        must_inc = case.get("must_include", [])
        must_cite_tags_any = case.get("must_cite_tags_any", [])
        max_latency = int(case.get("max_latency_ms", 3000))

        try:
            resp = call_chat(args.base, args.key, prompt)
        except Exception as ex:
            fail += 1
            details.append(f"[{cid}] CALL-ERR: {ex}")
            continue

        missing_keys = [k for k in REQUIRED_KEYS if k not in resp]
        if missing_keys:
            fail += 1
            details.append(f"[{cid}] SCHEMA-ERR missing keys: {missing_keys}")
            continue

        ans = resp.get("answer","") or resp.get("data",{}).get("answer","")
        if not isinstance(ans, str):
            fail += 1
            details.append(f"[{cid}] SCHEMA-ERR answer not string")
            continue

        # must include checks
        miss = [k for k in must_inc if k.lower() not in (ans or "").lower()]
        if miss:
            fail += 1
            details.append(f"[{cid}] ANSWER-ERR missing keyphrases: {miss} ; answer: {ans[:200]}")
            continue

        # citations contain at least one expected tag (any)
        if must_cite_tags_any:
            if not has_any_citation_tag(resp.get("citations",[]), must_cite_tags_any):
                fail += 1
                details.append(f"[{cid}] CITE-ERR expected any of tags={must_cite_tags_any} not found in citations")
                continue

        # latency check
        lat = int(resp.get("metrics",{}).get("latency_ms", 999999))
        if lat > max_latency:
            fail += 1
            details.append(f"[{cid}] PERF-ERR latency {lat}ms > {max_latency}ms")
            continue

        ok += 1

    print(f"PASS: {ok} / {len(cases)} ; FAIL: {fail}")
    for d in details[:20]:
        print(" -", d)
    if fail > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
