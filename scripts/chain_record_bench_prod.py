import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from chain_record_bench import bench
if __name__ == "__main__":
    import json
    out = bench()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)
