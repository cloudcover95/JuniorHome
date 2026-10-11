import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from anticon_vote import secure
if __name__ == "__main__":
    room = ["home", "pi-1", "phone"]
    print(json.dumps({"user": secure("dev", "user", room, ["home", "pi-1"]),
                      "bot": secure("", "user", room, ["home"])}, indent=2))
