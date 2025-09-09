# ingest/simhash.py
import re, hashlib

_WORD = re.compile(r"\w+", re.UNICODE)

def _h64(token: str) -> int:
    # Stable 64-bit hash from md5 (first 8 bytes)
    d = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)

def simhash64(text: str) -> int:
    tokens = _WORD.findall((text or "").lower())
    if not tokens:
        return 0
    v = [0]*64
    for t in tokens:
        h = _h64(t)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return ((a ^ b).bit_count())
