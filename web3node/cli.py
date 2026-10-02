"""Home CLI. One verb, one file. Cross-platform stdlib."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
SURFACES = ("code", "python", "blender", "llm")
KERNELS = ("cpu", "mlx", "cuda", "vulkan", "asahi")


def dispatch(argv: list[str]) -> dict:
    verb = argv[0] if argv else "help"
    if verb in SURFACES or verb in KERNELS:
        OS.mkdir(parents=True, exist_ok=True)
        kind = "surface" if verb in SURFACES else "kernel"
        body = {
            "kind": kind,
            "name": verb,
            "launch": False,
            "fetch_driver": False,
            "bpy": False,
            "model_pull": False,
            "writes": 1,
            "bind": "127.0.0.1",
        }
        (OS / f"{kind}_{verb}.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
        return body
    return {
        "cmd": "help",
        "surfaces": list(SURFACES),
        "kernels": list(KERNELS),
        "boot": False,
    }


if __name__ == "__main__":
    print(json.dumps(dispatch(sys.argv[1:]), indent=2))
