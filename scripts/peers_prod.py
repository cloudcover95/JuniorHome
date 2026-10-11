import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from peers import agree
if __name__ == "__main__":
    print(json.dumps(agree([{"name": "home", "vote": True}, {"name": "pi-1", "vote": True}]), indent=2))
