import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from render_wrap import wrap
if __name__ == "__main__":
    print(json.dumps({"open": wrap("open"), "t4": wrap("t4")}, indent=2))
