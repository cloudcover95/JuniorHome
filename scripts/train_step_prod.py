import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web3node"))
from train_step import step
if __name__ == "__main__":
    print(json.dumps(step("/no/such/model.gguf", False), indent=2))
