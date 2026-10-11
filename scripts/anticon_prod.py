import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from anticon import fold
if __name__ == "__main__":
    print(json.dumps({"ok": fold("dev", ["home"], ["gaia", "home"]),
                      "bot": fold("", ["home"], ["gaia"]),
                      "burst": fold("dev", ["home"], ["gaia"] * 40)}, indent=2))
