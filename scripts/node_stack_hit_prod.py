import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from node_stack_hit import stack, node_hit
if __name__ == "__main__":
    print(json.dumps({"stack": stack(["notes"]), "hit": node_hit("gaia"), "miss": node_hit("nope")}, indent=2))
