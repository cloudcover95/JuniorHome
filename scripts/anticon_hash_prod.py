import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from anticon_hash import guard
if __name__ == "__main__":
    print(json.dumps({"ok": guard("dev", ["gaia", "home"]), "bot": guard("", ["gaia"])}, indent=2))
