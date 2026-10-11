import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from preserve import drop
if __name__ == "__main__":
    print(json.dumps(drop("dev", {"gaia": 2, "home": 1}), indent=2))
