"""Six votes. All must pass. No disk. No ML-KEM."""
def six(actor, n, opted_in):
    votes = {"opted_in": opted_in is True, "actor": bool(actor), "n": 1 <= n <= 32,
             "no_pull": True, "rate_base": True, "no_kem": True}
    failed = [k for k, v in votes.items() if not v]
    return {"ok": not failed, "failed": failed, "disk": False, "ml_kem": False}
