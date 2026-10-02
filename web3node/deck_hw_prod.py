"""Production admit for a class node. Does not open the device."""
from pathlib import Path
from deck_native import probe
from write_gate import commit
OUT = Path.home() / ".juniorhome" / "deck" / "hw_prod.txt"
CLOSED = {"t4", "host"}
def allow(env="t4"):
    row = probe()
    integrate = bool(row["present"] and env in CLOSED)
    text = f"integrate={int(integrate)} present={int(row['present'])}"
    if env not in CLOSED:
        written = {"disk": False, "why": "open"}
    elif not row["present"]:
        written = {"disk": False, "why": "no_node"}
    else:
        written = commit(env, OUT, text)
    return {"integrate": integrate, "present": row["present"], "opened": False,
            "live": False, "env": env, "disk": written["disk"], "why": written["why"],
            "model_pull": False}
