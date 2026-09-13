"""Five JuniorOSai persistent-homology apps."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from home_terraform import inject
from ph_algorithms import h0_persistence
from tda_kit import infer

def _book():
    path = Path(__file__).resolve().parent / "vault" / "live_book.json"
    if not path.is_file():
        return None
    close = np.asarray(json.loads(path.read_text(encoding="utf-8")).get("close") or [], dtype=np.float64)
    if close.size < 8:
        return None
    return np.diff(np.log(np.clip(close, 1e-9, None)), axis=1)[:24, :8]

def _dem():
    ply = Path(__file__).resolve().parent / "vault" / "dem" / "flagstaff.ply"
    if not ply.is_file():
        return None
    pts = []
    for line in ply.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 3:
            try:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            except ValueError:
                pass
    return np.asarray(pts) if pts else None

def app_components(points):
    row = infer(points, "flagstaff field components")
    return {"app": "components", "beta0": row["beta0"], "edges": row["edges"], "port": row["terraform"]["port"]}

def app_long_bars(points):
    h0 = h0_persistence(points)
    deaths = np.asarray([p[1] for p in (h0.get("pairs_head") or [])], dtype=np.float64)
    if deaths.size == 0:
        deaths = np.asarray([float(h0.get("mean_death") or 0.0)])
    med = float(np.median(deaths))
    return {"app": "long_bars", "n_signal": int((deaths > med).sum()), "n_noise": int((deaths <= med).sum()), "median_death": med}

def app_merge_scale(points):
    h0 = h0_persistence(points)
    return {"app": "merge_scale", "last_death": h0.get("max_death"), "infinite_bars": h0.get("infinite_bars")}

def app_terrain(_points):
    pts = _dem()
    if pts is None:
        return {"app": "terrain", "ok": False}
    row = infer(pts, "flagstaff stonefield terrain")
    return {"app": "terrain", "ok": True, "beta0": row["beta0"], "edges": row["edges"], "n": row["n"]}

def app_route(points):
    row = infer(points, "flagstaff field route")
    connected = int(row["beta0"] or 0) <= 1
    return {"app": "route", "connected": connected, "port": "JuniorBitNetFieldCore" if connected else "JuniorOSai"}

def run_all():
    pts = _book()
    source = "live_book_returns" if pts is not None else "gaussian"
    if pts is None:
        pts = np.random.default_rng(11).normal(size=(20, 3))
    apps = {"components": app_components(pts), "long_bars": app_long_bars(pts),
            "merge_scale": app_merge_scale(pts), "terrain": app_terrain(pts), "route": app_route(pts)}
    tf = inject("juniorosai ph apps flagstaff")
    out = {"source": source, "apps": apps, "terraform": {k: tf.get(k) for k in ("port", "ok")}}
    path = Path(__file__).resolve().parent / "vault" / "ph_apps.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    out["out"] = str(path)
    return out

if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, default=str))
