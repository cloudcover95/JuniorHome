import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from os_mesh import mesh
SLOTS = [{"name": "overnight", "when": "daily 01:00"}, {"name": "daily-audit", "when": "daily 16:45"},
         {"name": "slot-19", "when": "daily 19:00"}, {"name": "weekly", "when": "Mon 17:00"},
         {"name": "omega-obj", "when": "Sat 10:00"}, {"name": "junioros-core", "when": "Sat 11:00"}]
def workflow(env="t4"):
    row = mesh("deck audio", env)
    out = {"slots": SLOTS, "missing": ["07:00", "13:00"], "mesh_sha3": row["sha3"],
           "hops": row["hops"], "model_pull": False, "bind": "127.0.0.1"}
    dest = Path.home() / ".juniorhome" / "gaia_mesh" / "workflow.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
if __name__ == "__main__":
    print(json.dumps(workflow(), indent=2))
