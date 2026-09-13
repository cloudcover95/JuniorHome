"""Robot stack on JuniorCloud Home."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from bitnet_orig import absmean as absmean_list
from fieldcore_bridge import pick_juniorllm_port
from home_kernel import dispatch
from home_terraform import inject
from second_brain import write_note
from trit_cache import compare, pack_trits
from xr_scene import scene as xr_scene

def _depth(n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    return (np.sin(xx / 8.0) + np.cos(yy / 9.0)).astype(np.float64)

def stack():
    depth = _depth()
    q, g = absmean_list(depth.reshape(-1).tolist())
    ticket = pack_trits(np.clip(np.rint(np.asarray(q)), -1, 1))
    sizes = compare(np.clip(np.rint(depth / (np.mean(np.abs(depth)) + 1e-9)), -1, 1))
    tf = inject("robot lidar omega field flagstaff")
    port = pick_juniorllm_port("robot field lidar omega")
    ker = dispatch("offline dem lidar robot", domain="terrain", watts=12)
    xr = xr_scene(12)
    note = write_note(Path(__file__).resolve().parent / "vault",
                      f"# robot stack\nticket={len(ticket)}B port={port} hid=crispy-mouse\n")
    row = {"pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "source": "sim"},
           "ticket_bytes": len(ticket), "gamma": g, "sizes": sizes,
           "recognize": {"kind": "trit-signature"}, "llm": port,
           "terraform": {k: tf.get(k) for k in ("port", "ok")},
           "kernel": {k: ker.get(k) for k in ("profile", "surface", "ue_boot")},
           "omega_xr": {k: xr.get(k) for k in ("quant", "ue_boot", "frameworks")},
           "hid": "crispy-mouse", "memsys": str(note), "ue_boot": False}
    out = Path(__file__).resolve().parent / "vault" / "robot_stack.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(stack(), indent=2, default=str))
