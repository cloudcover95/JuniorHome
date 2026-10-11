import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from poll import poll
if __name__ == "__main__":
    print(json.dumps({"user": poll("dev", "user", ["yes", "yes"]), "node": poll("home", "node", ["yes"])}, indent=2))
