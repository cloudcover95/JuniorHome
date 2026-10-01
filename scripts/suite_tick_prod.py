#!/usr/bin/env python3
"""One Home tick: audio inbox, then trit join. No model pull."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "web3node"
sys.path.insert(0, str(root))
from audio_digest import scan
from osai_join import join

if __name__ == "__main__":
    audio = scan()
    body = join()
    print(json.dumps({"audio_n": audio.get("n", 0), "join": body}, indent=2))
