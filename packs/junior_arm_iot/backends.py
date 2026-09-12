"""Kernel-domain router. Same AbsMean on stdlib, mlx, cuda, arm, asahi."""
from __future__ import annotations
import os, platform, importlib
from . import quant
DOMAINS = ("stdlib", "mlx", "cuda", "arm", "asahi")

def detect():
    mach = platform.machine().lower(); sysname = platform.system().lower()
    arm = mach in ("arm64", "aarch64", "armv8", "armv7l")
    asahi = arm and sysname == "linux"
    mlx = False; cuda = False
    try:
        importlib.import_module("mlx.core"); mlx = True
    except Exception:
        pass
    try:
        torch = importlib.import_module("torch"); cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    pref = os.environ.get("JUNIOR_BITNET_BACKEND", "").strip().lower()
    if pref not in DOMAINS:
        pref = "mlx" if mlx else "cuda" if cuda else "asahi" if asahi else "arm" if arm else "stdlib"
    return {"preferred": pref, "arm": arm, "asahi": asahi, "mlx": mlx, "cuda": cuda, "machine": mach, "system": sysname}

def absmean_quantize(xs, backend=None):
    info = detect(); name = (backend or info["preferred"]).lower()
    if name not in DOMAINS: name = "stdlib"
    if name == "mlx" and info["mlx"]:
        mx = importlib.import_module("mlx.core")
        w = mx.array(xs, dtype=mx.float32); scale = mx.mean(mx.abs(w))
        s = float(scale.item()) if hasattr(scale, "item") else float(scale)
        if s <= 1e-12: return [0]*len(xs), 0.0, "mlx"
        scaled = w / scale; q = mx.where(scaled > 0.5, 1, mx.where(scaled < -0.5, -1, 0))
        return [int(v) for v in q.tolist()], s, "mlx"
    if name == "cuda" and info["cuda"]:
        torch = importlib.import_module("torch")
        w = torch.tensor(xs, dtype=torch.float32, device="cuda"); scale = torch.mean(torch.abs(w)); s = float(scale.item())
        if s <= 1e-12: return [0]*len(xs), 0.0, "cuda"
        scaled = w / scale; q = torch.where(scaled > 0.5, 1, torch.where(scaled < -0.5, -1, 0))
        return [int(v) for v in q.detach().cpu().tolist()], s, "cuda"
    trits, scale = quant.quant_vec(xs)
    tag = name if name in ("arm", "asahi", "stdlib") else "stdlib"
    return trits, scale, tag
