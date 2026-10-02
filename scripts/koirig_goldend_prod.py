import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from koirig_goldend import flip
if __name__ == "__main__":
    print(json.dumps({"filter": flip("filter"), "gate": flip("gate")}, indent=2))
