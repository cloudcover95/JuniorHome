import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from yield_pool import yield_of
if __name__ == "__main__":
    print(json.dumps(yield_of("dev", 5, [0.1, 0.1, 0.3]), indent=2))
