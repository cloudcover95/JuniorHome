import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from gated_pack import close
if __name__ == "__main__":
    print(json.dumps(close("dev", ["gaia", "home"], True), indent=2))
    sys.exit(0)
