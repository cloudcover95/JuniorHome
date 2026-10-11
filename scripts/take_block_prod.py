import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from take_block import block
if __name__ == "__main__":
    print(json.dumps({"open": block("dev", ["gaia", "home"], [0.5], False),
                      "cold": block("dev", ["gaia"], [0.5], True)}, indent=2))
