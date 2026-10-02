import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from mem_bench import bench
if __name__ == "__main__":
    print(json.dumps({"open": bench("open"), "t4": bench("t4")}, indent=2))
