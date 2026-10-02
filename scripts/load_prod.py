import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from load import load
if __name__ == "__main__":
    print(json.dumps(load("/no/such/model.gguf", False), indent=2))
