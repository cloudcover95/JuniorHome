import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from tip_sign import sign_tip
if __name__ == "__main__":
    out = sign_tip()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] and out["signs"] == 1 else 1)
