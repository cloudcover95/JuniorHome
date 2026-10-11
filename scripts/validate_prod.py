import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from validate import make, valid
if __name__ == "__main__":
    row = make("prompt|dev")
    bad = dict(row)
    bad["body"] = bytes([9]) + row["body"][1:]
    print(json.dumps({"ok": valid(row), "bad": valid(bad)}, indent=2))
