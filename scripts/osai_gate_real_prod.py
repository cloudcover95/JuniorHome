import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from osai_gate import gate
if __name__ == "__main__":
    ok = gate("dev", ["gaia", "home"], True)
    bot = gate("", ["gaia"], True)
    print(json.dumps({"ok": ok, "bot": bot}, indent=2))
    sys.exit(0 if ok["ok"] and not bot["ok"] else 1)
