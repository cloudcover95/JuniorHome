"""Write one note into every Home digest point. Stdlib. No model pull."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path.home() / ".juniorhome" / "digest"
POINTS = (
    "home", "os", "osai", "llm", "omega", "cad", "gaia", "goldend", "geode",
    "field", "deck", "audio", "web3", "stock", "sol", "vault", "stone",
    "climbs", "engr", "poker", "memsys", "quant", "fetch", "drive", "coach",
    "python", "forge", "cpu", "mlx", "cuda", "vulkan", "asahi", "wallet",
)


def digest(note: str = "JuniorHome") -> dict:
    ROOT.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha3_256(note.encode()).hexdigest()[:16]
    written = []
    for name in POINTS:
        path = ROOT / f"{name}.jsonl"
        row = {
            "protocol": "goldend-osai-omega/1",
            "point": name,
            "note": note[:160],
            "sha3": sha,
            "ts": int(time.time()),
            "model_pull": False,
            "bind": "127.0.0.1",
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        written.append(name)
    index = {"n": len(written), "points": written, "sha3": sha}
    (ROOT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index
