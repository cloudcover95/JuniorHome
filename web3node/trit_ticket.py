"""Trit ticket engine — WebXR payload + AbsMean pack."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from bitnet_orig import absmean
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject
from second_brain import write_note
from trit_cache import compare, pack_trits
from xr_scene import scene as xr_base

def ticket(n=16):
    base = xr_base(n)
    pts = np.asarray(base["points"], dtype=np.float64)
    q, gamma = absmean(pts.reshape(-1).tolist())
    packed = pack_trits(np.clip(np.rint(np.asarray(q)), -1, 1))
    sizes = compare(np.clip(np.rint(pts / (np.mean(np.abs(pts)) + 1e-9)), -1, 1))
    tf = inject("juniorosai xr trit ticket flagstaff")
    port = pick_juniorllm_port("juniorosai xr trit")
    note = write_note(Path(__file__).resolve().parent / "vault",
                      f"# trit ticket\n\nn={n} pack={len(packed)}B port={port}\n")
    payload = {"webxr": {"kind": "line-network", "points": base["points"], "edges": base["edges"]},
               "ticket": {"quant": "ternary-1.58", "gamma": gamma, "bytes": len(packed),
                          "hex": packed.hex(), "sizes": sizes},
               "engine": {"home": "trit_ticket", "llm": port,
                           "terraform": {k: tf.get(k) for k in ("port", "ok", "fusion_backend")},
                           "obsidian": str(note)},
               "frameworks": base["frameworks"], "not_used": base["not_used"]}
    out = Path(__file__).resolve().parent / "vault" / "xr" / "trit_ticket.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["out"] = str(out)
    return payload

if __name__ == "__main__":
    row = ticket()
    print(json.dumps({"edges": len(row["webxr"]["edges"]), "ticket_bytes": row["ticket"]["bytes"],
                      "port": row["engine"]["llm"]}, indent=2))
