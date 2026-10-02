import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from osai_caps import expand
if __name__ == "__main__":
    print(json.dumps({"gaia_t0": expand("gaia", "t0", False), "deck_t4": expand("deck", "t4", False)}, indent=2))
