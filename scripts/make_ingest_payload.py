#!/usr/bin/env python3
"""
Builds a /ingest/batch payload from all *.md files under ./wiki.
Usage:
  python scripts/make_ingest_payload.py <user_email>
Prints a single JSON object to stdout:
  {"items":[{user_email, type, text, tags}, ...]}
"""
import sys, json, os

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def tags_for(path: str):
    rel = path.replace("\\", "/")
    if rel.startswith("wiki/"):
        rel = rel[len("wiki/"):]
    # tag the path (slash -> underscore) for traceability
    path_tag = rel[:-3].replace("/", "_") if rel.endswith(".md") else rel
    return ["clickup", "wiki", path_tag]

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/make_ingest_payload.py <user_email>", file=sys.stderr)
        sys.exit(2)
    user_email = sys.argv[1]

    items = []
    for root, _, files in os.walk("wiki"):
        for fn in files:
            if not fn.lower().endswith(".md"):
                continue
            path = os.path.join(root, fn)
            text = read_text(path)
            items.append({
                "user_email": user_email,
                "type": "semantic",
                "text": text,
                "tags": tags_for(path),
            })

    print(json.dumps({"items": items}, ensure_ascii=False))

if __name__ == "__main__":
    main()
