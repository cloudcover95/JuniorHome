"""Base rate until a poll agrees."""
BASE = 0.05
def cloud_rate(agreed, packed):
    if not agreed or packed is None:
        return {"rate": BASE, "source": "base", "caller_set": False}
    return {"rate": round(min(0.2, max(0.01, packed)), 4), "source": "poll", "caller_set": False}
