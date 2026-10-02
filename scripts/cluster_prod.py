import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from cluster import cluster
if __name__ == "__main__":
    print(json.dumps(cluster(), indent=2))
