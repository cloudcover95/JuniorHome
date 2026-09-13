"""Checks-and-balance engineer. Live: JuniorLLM ports/flagstaff_balance.py."""
from __future__ import annotations
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from off_caps import all_off

def check(note: str, *, consent: bool = True, private: bool = False):
    try:
        import importlib
        return importlib.import_module("ports.flagstaff_balance").check(note, consent=consent, private=private)
    except Exception:
        tf = inject(note)
        port = pick_juniorllm_port(note)
        votes = {"terraform_ok": bool(tf.get("ok")), "covenant": not (private and not consent),
                 "junior_port": str(port).startswith("Junior"),
                 "no_bind": "0.0.0.0" not in str(tf.get("text") or ""),
                 "off_rails": all(v.get("state") in {"off", "allowed"} for v in all_off().values())}
        return {"ok": all(votes.values()), "votes": votes, "port": port, "source": "home-twin"}
