import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_engine import run
if __name__ == "__main__":
    print(json.dumps({"open": run("open"), "t4": run("t4")}, indent=2))
