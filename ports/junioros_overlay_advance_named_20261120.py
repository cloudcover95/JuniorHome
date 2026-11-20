"""JuniorOS overlay advance — ROADMAP / BETA_TO_OS named item.

Advances JuniorHome docs/ROADMAP.md and JuniorLLM docs/BETA_TO_OS.md named item:
JuniorOS overlay. (bitnetd 127.0.0.1:8765 / i2sd 127.0.0.1:8767 / juniorctl).
Pure in-process. Declarative facts only.
Never opens a socket, never pulls a model, never fetches, never boots UE,
never imports juniorctl or bitnetd.

Base rate 0.05. Pool held. mint false. socket false. fetch false.
layer_one false. ue5_launch false. Bind 127.0.0.1 only.
Close rule: sha256(prev || pack || text)[:16]. Stdlib only.
No wrap of path.py, chain_record.py, or record_block.py.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
FETCH = False
LAYER_ONE = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
BITNETD = "127.0.0.1:8765"
I2SD = "127.0.0.1:8767"
JUNIORCTL = "juniorctl"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "JuniorOS overlay. (bitnetd 127.0.0.1:8765 / i2sd 127.0.0.1:8767 / juniorctl)"
DAY = "2026-11-20"
PROBE_ORIGIN = "operator"

def close(prev: str, pack: str, text: str) -> str:
    blob = (prev + pack + text).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prev", default=GENESIS)
    p.add_argument("--pack", default="junioros")
    p.add_argument("--text", default=ROADMAP_ITEM)
    args = p.parse_args()
    tip = close(args.prev, args.pack, args.text)
    print(f"rate={RATE} pool_held={POOL_HELD} mint={MINT} socket={SOCKET}")
    print(f"named={ROADMAP_ITEM}")
    print(f"close={tip}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
