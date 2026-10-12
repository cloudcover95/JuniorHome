import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from tip_next import next_close
if __name__ == "__main__":
    out = next_close()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)
