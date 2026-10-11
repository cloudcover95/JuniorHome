import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from cold_take import take_cold
if __name__ == "__main__":
    print(json.dumps({"cold": take_cold("dev", ["gaia"], [0.5], True),
                      "open": take_cold("dev", ["gaia", "home"], [0.5], False)}, indent=2))
