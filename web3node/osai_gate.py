"""The named gate was six Trues. This one can fail. No mesh write."""
from gated_pack import close
def gate(actor="dev", hits=None, opted_in=True):
    row = close(actor, hits or ["gaia"], opted_in)
    return {"ok": row["ok"], "why": row.get("why"), "hash": row.get("hash"),
            "hardcoded": False, "disk": False, "ml_kem": False, "live": False}
