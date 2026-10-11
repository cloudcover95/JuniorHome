import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from trit_receipt import receipt, verify
if __name__ == "__main__":
    secret = b"local-only"
    row = receipt("4641cb306b59165b", "cu-aurora share", secret)
    print(json.dumps({"row": row, "ok": verify(row, "cu-aurora share", secret), "bad": verify(row, "other", secret)}, indent=2))
