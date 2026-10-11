import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from optin import run
if __name__ == "__main__":
    print(json.dumps({"off": run(False), "on": run(True)}, indent=2))
