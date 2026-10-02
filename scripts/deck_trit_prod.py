import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_trit import ticket
if __name__ == "__main__":
    print(json.dumps({"t4": ticket("t4"), "t0": ticket("t0")}, indent=2))
