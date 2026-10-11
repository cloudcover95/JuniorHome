import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from base_rate import cloud_rate
if __name__ == "__main__":
    print(json.dumps({"base": cloud_rate(False, None), "poll": cloud_rate(True, 0.08)}, indent=2))
