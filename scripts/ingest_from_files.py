import os, sys, glob, re
from typing import List
from pypdf import PdfReader
from agent.ingest import distill_chunk, chunk_text

def read_text(path: str) -> str:
    if path.lower().endswith(".md") or path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path}")

def main():
    user_id = os.environ.get("INGEST_USER_ID","00000000-0000-0000-0000-000000000000")
    base_tags = os.environ.get("DEFAULT_INGEST_TAGS","docs").split(",")
    paths = sys.argv[1:] or glob.glob("docs/**/*.*", recursive=True)
    all_ids = []
    for p in paths:
        text = read_text(p)
        for ch in chunk_text(text):
            ids = distill_chunk(user_id=user_id, raw_text=ch, base_tags=base_tags, make_qa=True)
            all_ids.extend(ids)
    print("Created memories:", len(all_ids))

if __name__ == "__main__":
    main()
