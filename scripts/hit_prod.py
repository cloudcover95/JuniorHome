import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from hit import hit
if __name__ == "__main__":
    nodes = [{"name": "pi-1", "host": True, "hosts": ["share"]}]
    print(json.dumps(hit("share", nodes), indent=2))
