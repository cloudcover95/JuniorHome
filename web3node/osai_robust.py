"""OSai gate. Open envelope does not touch disk."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from trit_energy import rate
from write_gate import commit
MESH = Path.home() / ".juniorhome" / "gaia_mesh" / "os_mesh.jsonl"
OUT = Path.home() / ".juniorhome" / "gaia_mesh" / "osai_robust.jsonl"
def _prior():
    if not MESH.exists():
        return ""
    lines = MESH.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return ""
    return str(json.loads(lines[-1]).get("sha3") or "")
def gate(note="JuniorOSai", env="open"):
    row = rate(note, env)
    sha = hashlib.sha3_256(note.encode()).hexdigest()[:16]
    prior = _prior()
    votes = {"protocol": row["protocol"] == "goldend-osai-omega/1",
             "bind": row["bind"] == "127.0.0.1",
             "no_pull": row["model_pull"] is False,
             "energy": row["band"] == "pass",
             "n": row["n"] > 0,
             "sha3": bool(prior) and prior == sha}
    ok = all(votes.values())
    line = json.dumps({"ok": ok, "band": row["band"], "sha3": sha}) + "\n"
    written = commit(env, OUT, line)
    return {"ok": ok, "allow_push": ok, "votes": votes, "energy": row["energy"],
            "band": row["band"], "sha3": sha, "prior": prior, "env": env,
            "disk": written["disk"], "why": written["why"],
            "bind": "127.0.0.1", "model_pull": False}
