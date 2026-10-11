import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from complete import complete
if __name__ == "__main__":
    print(json.dumps(complete("dev", {"prompt": 4, "gaia": 1}), indent=2))
