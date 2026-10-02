import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from trit_energy import rate
if __name__ == "__main__":
    print(json.dumps(rate(), indent=2))
