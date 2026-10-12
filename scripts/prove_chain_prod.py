import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from prove_chain import prove
if __name__ == "__main__":
    row = prove([("dev", ["gaia"]), ("home", ["prompt"]), ("pi", ["field"])])
    print(json.dumps(row, indent=2))
    sys.exit(0 if row["ok"] else 1)
