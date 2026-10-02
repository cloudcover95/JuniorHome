import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from deck_hw_prod import allow
if __name__ == "__main__":
    print(json.dumps(allow("t4"), indent=2))
