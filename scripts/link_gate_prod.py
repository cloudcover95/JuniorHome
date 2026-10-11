import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from link_gate import link
if __name__ == "__main__":
    room = ["home", "pi-1"]
    print(json.dumps({"ok": link("home", room, "abc", "abc"), "spoof": link("stranger", room, "abc", "abc")}, indent=2))
