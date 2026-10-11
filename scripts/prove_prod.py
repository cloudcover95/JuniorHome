import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from offline import offline
from prove import prove
if __name__ == "__main__":
    row = offline(90)
    print(json.dumps(prove(90, row["tip"]), indent=2))
