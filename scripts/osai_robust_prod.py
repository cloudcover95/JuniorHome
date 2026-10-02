import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from osai_robust import gate
if __name__ == "__main__":
    body = gate()
    print(json.dumps(body, indent=2))
    sys.exit(0 if body["allow_push"] else 1)
