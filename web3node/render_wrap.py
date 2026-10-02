"""End wrap. Frame stays in RAM until the envelope closes. Not a framebuffer dump."""
from pathlib import Path
from write_gate import commit
OUT = Path.home() / ".juniorhome" / "deck" / "wrap.txt"
def frame(n=16):
    return bytes((i * 17) % 251 for i in range(n))
def wrap(env="open", n=16):
    raw = frame(n)
    written = commit(env, OUT, raw.hex())
    return {"bytes": n, "ram": not written["disk"], "disk": written["disk"],
            "why": written["why"], "released": written["disk"],
            "ue5_launch": False, "blender": False, "model_pull": False}
