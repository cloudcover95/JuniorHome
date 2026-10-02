import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from local_compute import backends
if __name__ == "__main__":
    print(json.dumps(backends(), indent=2))
