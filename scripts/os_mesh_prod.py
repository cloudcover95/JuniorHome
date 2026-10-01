import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from os_mesh import mesh
if __name__ == "__main__":
    print(json.dumps(mesh(), indent=2))
