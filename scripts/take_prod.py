import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from take import take
if __name__ == "__main__":
    print(json.dumps({"ok": take("dev", ["home"], ["gaia", "home"], [0.5]),
                      "bot": take("", ["home"], ["gaia"], [0.5])}, indent=2))
