"""AFR error → cell delta. Authority cap. User confirm required. No ECU write."""
from __future__ import annotations
import json
from typing import Any

AUTHORITY = 0.05
TRANSIENT = 0.2

def recommend(current: float, target: float, prev: float | None = None, authority: float = AUTHORITY) -> dict[str, Any]:
    err = float(target) - float(current)
    if prev is not None and abs(float(current) - float(prev)) > TRANSIENT:
        return {"delta": 0.0, "reason": "transient", "confirm": False}
    delta = max(-authority, min(authority, err * 0.1))
    return {"delta": round(delta, 5), "error": round(err, 5), "authority": authority, "confirm": False}

if __name__ == "__main__":
    print(json.dumps(recommend(14.2, 14.7, prev=14.1), indent=2))
