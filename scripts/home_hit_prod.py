import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from home_hit import action, ACTIONS
if __name__ == "__main__":
    print(json.dumps({k: action(k) for k in ACTIONS}, indent=2))
