import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from geode_func import select
if __name__ == "__main__":
    print(json.dumps({"note": select("note"), "gamma": select("gamma")}, indent=2))
