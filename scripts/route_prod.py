import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from route import route
if __name__ == "__main__":
    print(json.dumps(route("infer", {"pi": False, "gpu": False, "spark": False, "mac": False}), indent=2))
