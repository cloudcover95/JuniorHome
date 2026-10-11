"""Juniors floor. Five pieces. None of them is a network."""
import hashlib
def gossip(note):
    return {"ok": True, "note": note[:64], "peers": 0, "bind": "127.0.0.1", "sent": False}
def agree(votes):
    if not votes:
        return {"ok": False, "why": "no_peers", "consensus": False}
    return {"ok": all(votes), "n": len(votes), "consensus": all(votes), "bft": False}
def mldsa_verify(sig, msg, pk):
    return {"ok": False, "why": "no_verifier", "pq_ready": False, "ml_dsa": False}
def block(rows):
    body = "\n".join(rows).encode()
    return {"n": len(rows), "hash": hashlib.sha256(body).hexdigest()[:16],
            "prev": "0" * 16, "produced": True, "peers": 0}
def dormant(last_seen, holder):
    frozen = last_seen is None or not holder
    return {"frozen": frozen, "sweep": False, "migrate": holder and not frozen,
            "hard_fork": True, "why": "holder_must_present" if frozen else "active"}
def floor():
    return {"gossip": gossip("share"), "consensus": agree([]), "mldsa": mldsa_verify(b"", b"", b""),
            "block": block(["cu-aurora"]), "dormant": dormant(None, False),
            "chain": False, "model_pull": False}
