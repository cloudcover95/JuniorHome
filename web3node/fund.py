"""Rate is the base. A poll only releases the pool."""
BASE = 0.05
def fund(pool, agreed):
    if not agreed:
        return {"rate": BASE, "pool": pool, "released": 0, "why": "held"}
    return {"rate": BASE, "pool": pool, "released": pool, "why": "poll"}
