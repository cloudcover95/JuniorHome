import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from pio_sm import run
if __name__ == "__main__":
    print(json.dumps({"t4": run("t4"), "t0": run("t0")}, indent=2))
