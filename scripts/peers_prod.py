import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from peers import peers, vote, majority
if __name__ == "__main__":
    room = peers(["home", "pi-1", "phone"])["peers"]
    votes = [vote("home", True, room), vote("pi-1", True, room), vote("stranger", True, room)]
    print(json.dumps({"room": room, "majority": majority(votes, room)}, indent=2))
