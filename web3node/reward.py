"""Reward comes from a completed close. Rate is fixed."""
RATE = 0.05
def reward(hits, host):
    if hits < 1:
        return {"ok": False, "why": "no_close", "juniors": 0}
    cloud = round(hits * RATE, 4)
    host_share = round((hits - cloud) / 2, 4) if host else 0
    return {"ok": True, "hits": hits, "juniors": cloud, "host_share": host_share,
            "rate": RATE, "caller_set": False, "mint": False}
