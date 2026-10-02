"""OSai robustness gate. Six AND. Fail closed. Not a model."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from trit_energy import rate
OUT = Path.home() / ".juniorhome" / "gaia_mesh" / "osai_robust.jsonl"
def gate(note="JuniorOSai"):
    row = rate(note)
    sha = hashlib.sha3_256(note.encode()).hexdigest()[:16]
    votes = {
        "protocol": row["protocol"] == "goldend-osai-omega/1",
        "bind": row["bind"] == "127.0.0.1",
        "no_pull": row["model_pull"] is False,
        "energy": row["band"] != "fail",
        "n": row["n"] > 0,
        "sha3": bool(sha),
    }
    body = {"ok": all(votes.values()), "votes": votes, "energy": row["energy"],
            "band": row["band"], "sha3": sha, "bind": "127.0.0.1", "model_pull": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ok": body["ok"], "band": body["band"], "sha3": sha}) + "\n")
    return body
