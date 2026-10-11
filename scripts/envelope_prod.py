import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from envelope import envelope
if __name__ == "__main__":
    print(json.dumps(envelope("dev", {"prompt": 4, "gaia": 1}, True, True), indent=2))
