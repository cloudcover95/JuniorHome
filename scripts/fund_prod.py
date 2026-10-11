import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from fund import fund
if __name__ == "__main__":
    print(json.dumps({"held": fund(1.5, False), "open": fund(1.5, True)}, indent=2))
