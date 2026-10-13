"""Local tip rule. Longest valid walk wins. No peers."""
from __future__ import annotations

import json
from pathlib import Path

from juniors_ledger import walk
from validate import make, valid

OUT = Path.home() / ".juniorhome" / "os" / "juniors_consensus.json"


def choose(candidates: list[dict]) -> dict:
    good = []
    prev_ok = "0" * 16
    for row in candidates:
        check = valid(row)
        if not check["ok"]:
            continue
        if row.get("prev") != prev_ok and prev_ok != "0" * 16:
            # fail closed on broken link after genesis
            return {"ok": False, "why": "link"}
        good.append(row)
        prev_ok = row["hash"]
    if not good:
        return {"ok": False, "why": "none"}
    tip = good[-1]
    return {"ok": True, "tip": tip["hash"], "n": len(good), "rule": "recompute"}


def safeguard() -> dict:
    genesis = make("genesis|0.05|0")
    second = make("home|0.05|1", genesis["hash"])
    fork = {"prev": "f" * 16, "hash": "ab" * 8, "body": b"\x01\x01\x01"}
    broken = {"prev": "b" * 16, "hash": second["hash"], "body": second.get("body", ""), "text": second.get("text", "")}
    picked = choose([genesis, second, fork])
    bad_link = choose([genesis, broken])
    walked = walk()
    body = {
        "ok": picked["ok"] and picked["tip"] == second["hash"] and not valid(fork)["ok"] and not bad_link["ok"],
        "tip": picked["tip"],
        "rule": "recompute",
        "ledger_tip": walked.get("tip"),
        "peers": 0,
        "listen": False,
        "mint": False,
        "socket": False,
        "network": False,
        "layer_one": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
