#!/usr/bin/env python3
import os, sys, time, json, requests

BASE = os.getenv("BASE_URL", "https://suaps-brain.onrender.com")
KEY  = os.getenv("X_API_KEY")

def call_chat(prompt):
    t0 = time.time()
    r = requests.post(
        f"{BASE}/chat",
        headers={"Content-Type":"application/json","X-API-Key": KEY or ""},
        json={"prompt": prompt, "role":"researcher", "debug": False},
        timeout=40
    )
    dt = int((time.time()-t0)*1000)
    return r.status_code, dt, r.text

def main(path):
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    fails = 0
    for c in cases:
        print(f"\n==> {c['id']} :: {c['prompt']}")
        code, ms, body = call_chat(c["prompt"])
        print(f"HTTP {code} in {ms}ms")
        if code != 200:
            print("FAIL: non-200")
            fails += 1
            continue
        try:
            data = json.loads(body)
        except Exception as e:
            print("FAIL: invalid JSON:", e)
            fails += 1
            continue
        ans = (data.get("answer") or "").lower()
        cits = data.get("citations") or []
        if ms > c.get("max_latency_ms", 999999):
            print("FAIL: latency too high")
            fails += 1
        ok_terms = all(term.lower() in ans for term in c.get("must_include", []))
        ok_cites = any(any(token in (str(x) or "") for token in c.get("must_cite_any", [])) for x in cits) if c.get("must_cite_any") else True
        if not ok_terms:
            print("FAIL: missing must_include terms")
            print("Answer:", ans[:400])
            fails += 1
        if not ok_cites:
            print("FAIL: citations missing expected tokens; got:", cits)
            fails += 1
        if fails == 0:
            print("PASS")
    if fails:
        print(f"\n{fails} failing case(s)")
        sys.exit(1)
    print("\nAll cases passed.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tests/evals/golden_set.json"
    main(path)
