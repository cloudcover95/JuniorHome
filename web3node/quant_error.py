"""Quant error + passive check."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from agent_stack import run as agents
from bitnet_orig import absmean
from fieldcore_bridge import pick_juniorllm_port
from home_kernel import dispatch
from home_terraform import inject
from int4_store import pack_int4, to_int4
from off_caps import refetch
from trit_cache import pack_trits

def _err(src, rec):
    d = src.reshape(-1) - rec.reshape(-1)
    n = float(np.linalg.norm(src) or 1.0)
    return {"mse": float(np.mean(d*d)), "max_abs": float(np.max(np.abs(d))), "rel_l2": float(np.linalg.norm(d)/n)}

def errors(n=128):
    x = np.random.default_rng(3).normal(size=n)
    qt, g = absmean(x.tolist())
    rec_t = np.asarray(qt, dtype=np.float64) * g
    q4 = to_int4(x)
    rec_4 = q4.astype(np.float64) / 7.0 * (float(np.max(np.abs(x))) or 1.0)
    pnl = None
    book = Path(__file__).resolve().parent / "vault" / "live_book.json"
    if book.is_file():
        raw = json.loads(book.read_text(encoding="utf-8"))
        closes = raw.get("close") or raw.get("C")
        if isinstance(closes, list) and len(closes) > 2:
            r = np.diff(np.log(np.clip(np.asarray(closes, dtype=np.float64), 1e-9, None)))
            pnl = {"n": int(r.size), "mean": float(np.mean(r)), "vol": float(np.std(r))}
    return {"n": n, "trit": {**_err(x, rec_t), "bytes": len(pack_trits(np.clip(np.rint(np.asarray(qt)), -1, 1)))},
            "int4": {**_err(x, rec_4), "bytes": len(pack_int4(q4))},
            "gguf": {"q4_0_block": {"weights": 32, "bytes": 18}, "q4_k": "not parsed"},
            "pnl_from_book": pnl, "refetch": refetch("yahoo")}

def check():
    row = errors()
    ag = agents("juniorosai field flagstaff domain-agnostic")
    assert row["trit"]["rel_l2"] < 1.0 and refetch("yahoo").get("ran") is False
    return {"ok": True, "port": pick_juniorllm_port("juniorosai field quant"),
            "tf": inject("juniorosai field quant error flagstaff").get("port"),
            "agents": ag.get("ok"), "ue_boot": dispatch("home clock", domain="agent", watts=45).get("ue_boot"),
            "errors": row}

if __name__ == "__main__":
    print(json.dumps(check(), indent=2, default=str))
