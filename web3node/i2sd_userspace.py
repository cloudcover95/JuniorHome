"""ROADMAP #3 i2sd userspace. 127.0.0.1:8767. No docker.sock."""
from __future__ import annotations

import hashlib

RATE = 0.05
BIND = "127.0.0.1:8767"
ITEM = "ROADMAP #3 i2sd userspace"


def close(prev: str, text: str) -> str:
    body = f"{prev}|{text}|{RATE}"
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def receipt(prev: str = "0" * 16, docker_sock: bool = False) -> dict:
    if docker_sock:
        return {"ok": False, "why": "docker.sock", "item": ITEM}
    if not BIND.startswith("127.0.0.1"):
        return {"ok": False, "why": "bind", "item": ITEM}
    tip = close(prev, BIND)
    return {
        "ok": True,
        "item": ITEM,
        "bind": BIND,
        "docker_sock": False,
        "rate": RATE,
        "hash": tip,
        "mint": False,
        "socket": False,
        "pool": "held",
    }


def selftest() -> int:
    row = receipt()
    bad = receipt(docker_sock=True)
    assert row["ok"] and row["rate"] == 0.05
    assert row["bind"] == "127.0.0.1:8767"
    assert not bad["ok"] and bad["why"] == "docker.sock"
    return 0


if __name__ == "__main__":
    import json
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(json.dumps(receipt(), indent=2))
