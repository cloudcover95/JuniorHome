import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from terraform_env import envelope
if __name__ == "__main__":
    print(json.dumps({"t4": envelope("t4"), "t0": envelope("t0")}, indent=2))
