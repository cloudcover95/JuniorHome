import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from take_receipt import receipt
if __name__ == "__main__":
    print(json.dumps({"open": receipt("dev", ["gaia", "home"], [0.5], False),
                      "cold": receipt("dev", ["gaia"], [0.5], True)}, indent=2))
