"""Home CLI. One verb, one file. Status does not write."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OS = Path.home() / ".juniorhome" / "os"
SURFACES = ("code", "python", "blender", "llm")
KERNELS = ("cpu", "mlx", "cuda", "vulkan", "asahi")


def status() -> dict:
    return {
        "cmd": "status",
        "surfaces": {name: (OS / f"surface_{name}.json").exists() for name in SURFACES},
        "kernels": {name: (OS / f"kernel_{name}.json").exists() for name in KERNELS},
        "app": (OS / "mobile.json").exists(),
        "boot": False,
        "writes": 0,
    }


def dispatch(argv: list[str]) -> dict:
    verb = argv[0] if argv else "help"
    if verb == "status":
        return status()
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
    return {"cmd": "help", "verbs": ["status", *SURFACES, *KERNELS], "boot": False}


if __name__ == "__main__":
    print(json.dumps(dispatch(sys.argv[1:]), indent=2))
