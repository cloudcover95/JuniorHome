"""Local tip rule. Longest valid walk wins. No peers."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_ledger import walk
from validate import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_consensus.json"


def choose(candidates: list[dict]) -> dict:
    good = []
    for row in candidates:
        check = valid(row)
        if check["ok"]:
            good.append(row)
    if not good:
        return {"ok": False, "why": "none"}
    tip = good[-1]
    return {"ok": True, "tip": tip["hash"], "n": len(good), "rule": "recompute"}


def safeguard() -> dict:
    genesis = make("genesis|0.05|0")
    second = make("home|0.05|1", genesis["hash"])
    fork = {"prev": "f" * 16, "hash": "ab" * 8, "body": b"\x01\x01\x01"}
    picked = choose([genesis, second, fork])
    walked = walk()
    body = {
        "ok": picked["ok"] and picked["tip"] == second["hash"] and not valid(fork)["ok"],
        "tip": picked["tip"],
        "rule": "recompute",
        "ledger_tip": walked.get("tip"),
        "peers": 0,
        "listen": False,
        "mint": False,
        "socket": False,
        "network": False,
        "layer_one": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
