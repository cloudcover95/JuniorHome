import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from ardour_ext import write
if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
