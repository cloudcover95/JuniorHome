"""Home CLI. One verb, one file. Status does not write."""
from __future__ import annotations
import json, sys
from pathlib import Path
from pick import pick
from deck_cli import dispatch as deck_dispatch
OS = Path.home() / ".juniorhome" / "os"
SURFACES = ("code", "python", "blender", "llm")
KERNELS = ("cpu", "mlx", "cuda", "vulkan", "asahi")
CORES = ("JuniorLLM", "JuniorOSai", "AGI_SDK", "web3node", "JuniorStock", "JuniorPython-Suite", "JuniorOmega", "Gaia")
def status():
    active = {}
    path = OS / "active.json"
    if path.exists():
        active = json.loads(path.read_text(encoding="utf-8"))
    return {"cmd": "status", "active": active.get("active"),
            "surfaces": {n: (OS / f"surface_{n}.json").exists() for n in SURFACES},
            "kernels": {n: (OS / f"kernel_{n}.json").exists() for n in KERNELS},
            "registry": (OS / "registry.json").exists(), "boot": False, "writes": 0}
def registry(env="t4"):
    cap = 8 if env == "t4" else 16
    rows = [{"name": n, "download_gb": 0.0} for n in CORES[:cap]]
    body = {"cmd": "registry", "env": env, "cores": rows, "n": len(rows),
            "model_pull": False, "bind": "127.0.0.1", "writes": 1}
    OS.mkdir(parents=True, exist_ok=True)
    (OS / "registry.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
    return body
def dispatch(argv):
    verb = argv[0] if argv else "help"
    if verb == "deck":
        return deck_dispatch(argv[1:])
    if verb == "status":
        return status()
    if verb == "pick" and len(argv) > 1:
        return pick(argv[1])
    if verb == "registry":
        return registry(argv[1] if len(argv) > 1 else "t4")
    if verb in SURFACES or verb in KERNELS:
        OS.mkdir(parents=True, exist_ok=True)
        kind = "surface" if verb in SURFACES else "kernel"
        body = {"kind": kind, "name": verb, "launch": False, "fetch_driver": False,
                "bpy": False, "model_pull": False, "writes": 1, "bind": "127.0.0.1"}
        (OS / f"{kind}_{verb}.json").write_text(json.dumps(body) + "\n", encoding="utf-8")
        return body
    return {"cmd": "help", "verbs": ["status", "pick", "registry", "deck", *SURFACES, *KERNELS], "boot": False}
