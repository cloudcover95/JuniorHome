import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from juniors_pq import sign
if __name__ == "__main__":
    print(json.dumps(sign("cu-aurora share"), indent=2))
