import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_controls import control
if __name__ == "__main__":
    print(json.dumps(control("gamma"), indent=2))
