import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from juniors import open_account, add_signer
if __name__ == "__main__":
    acc = open_account("cu-aurora")
    print(json.dumps({"open": acc, "add": add_signer(acc, "placeholder")}, indent=2))
