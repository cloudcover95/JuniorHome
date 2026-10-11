"""Juniors. Credit-union floor. Address is not the signer. Not a chain."""
import hashlib
GOLDEND = "juniors"
MISSING = ("peer gossip", "consensus", "ml-dsa verify", "block production", "dormant-account policy")
def address(tenant):
    return hashlib.sha256(f"juniors|{tenant}".encode()).hexdigest()[:16]
def open_account(tenant):
    return {"goldend": GOLDEND, "address": address(tenant), "signer": None,
            "signer_type": None, "pq_ready": False, "missing": list(MISSING),
            "chain": False, "model_pull": False}
def add_signer(account, signer, kind="ml-dsa"):
    if kind != "ml-dsa":
        return {"ok": False, "why": "not_pq", "pq_ready": False}
    return {"ok": True, "goldend": GOLDEND, "address": account["address"],
            "signer": signer, "signer_type": kind, "pq_ready": False,
            "why": "no_verifier", "chain": False}
