import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from gate_chain import run
if __name__ == "__main__":
    good = run([("dev", ["gaia"]), ("home", ["prompt"])])
    bad = run([("dev", ["gaia"]), ("", ["prompt"])])
    print(json.dumps({"good": good, "bad": bad}, indent=2))
    sys.exit(0 if good["ok"] and not bad["ok"] else 1)
