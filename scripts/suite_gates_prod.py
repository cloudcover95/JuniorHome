import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from suite_gates import suite
if __name__ == "__main__":
    import json
    row = suite()
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["gates"] else 1)
