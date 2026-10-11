import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from juniors_65 import block
if __name__ == "__main__":
    print(json.dumps(block(b"cu-aurora share|obsidian|gaia"), indent=2))
