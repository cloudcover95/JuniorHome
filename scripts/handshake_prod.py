import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from handshake import enroll, hello, accept
if __name__ == "__main__":
    home = enroll("home")
    fake = enroll("home")
    book = {"home": home["pk"]}
    print(json.dumps({"good": accept(book, hello(home, "abc")), "spoof": accept(book, hello(fake, "abc"))}, indent=2))
