import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from prod_status import status
if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
