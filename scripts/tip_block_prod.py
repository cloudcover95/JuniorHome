import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from tip_block import next_block
if __name__ == "__main__":
    print(json.dumps(next_block(), indent=2))
