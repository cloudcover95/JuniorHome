import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from strip_check import check
if __name__ == "__main__":
    print(json.dumps(check("ui/deck.html", "flashed false MX 0-15"), indent=2))
