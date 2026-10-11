import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from done_block import block
if __name__ == "__main__":
    print(json.dumps({"prompt": block("prompt"), "nope": block("nope")}, indent=2))
