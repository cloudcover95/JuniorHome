import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from module_pipe import route
if __name__ == "__main__":
    print(json.dumps({"gaia": route("gaia"), "deck": route("deck")}, indent=2))
