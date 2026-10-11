import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from rail_block import block
if __name__ == "__main__":
    print(json.dumps({"ok": block("dev", ["home"], 3), "burst": block("dev", ["home"], 40)}, indent=2))
