import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from rate import rate_of
if __name__ == "__main__":
    print(json.dumps(rate_of(["home", "pi-1", "phone"], ["home", "pi-1"]), indent=2))
