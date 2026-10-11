"""Completed Home actions. Counting is not running."""
import hashlib
from cloud_share import share
ACTIONS = ("prompt", "drawing", "bot", "llm", "agent")
def done(kind):
    if kind not in ACTIONS:
        return {"ok": False, "why": "not_done", "ran": False}
    cut = share({"actor": 1, "home": 1})
    return {"ok": True, "kind": kind, "receipt": hashlib.sha256(kind.encode()).hexdigest()[:16],
            "cloud": cut["cloud"], "ran": False, "model_pull": False, "network": False}
