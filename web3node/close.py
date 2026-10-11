"""Hits are a counter. One trit block at close."""
import hashlib
def close(hits):
    n = sum(hits.values())
    body = hashlib.sha256(str(sorted(hits.items())).encode()).hexdigest()[:16]
    return {"hits": n, "packed_once": True, "hash": body, "per_hit_pack": False,
            "per_hit_sign": False, "network": False}
