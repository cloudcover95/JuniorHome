import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from session_open import session
from rails import allow
if __name__ == "__main__":
    print(json.dumps({"session": session("dev"), "ok": allow("dev", ["home"], 3),
                      "burst": allow("dev", ["home"], 40), "multi": allow("dev", ["home", "home"], 2)}, indent=2))
