import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from device import device, DEVICES
if __name__ == "__main__":
    print(json.dumps([device(k, "dev") for k in DEVICES], indent=2))
