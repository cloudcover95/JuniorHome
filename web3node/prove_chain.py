"""Recompute a multi-close chain. No authority."""
from chain_run import run
def prove(sessions):
    first = run(sessions)
    second = run(sessions)
    return {"ok": first["ok"] and first["tip"] == second["tip"], "tip": first.get("tip"),
            "n": first.get("n"), "authority": False, "socket": False}
