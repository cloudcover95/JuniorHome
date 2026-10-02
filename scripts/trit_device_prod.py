import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from trit_device import device
if __name__ == "__main__":
    print(json.dumps({"t4": device("t4"), "t0": device("t0")}, indent=2))
