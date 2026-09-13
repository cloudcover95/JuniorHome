"""10 m DEM tile via USGS EPQS. Not 1 m nationwide LiDAR."""
from __future__ import annotations
import json, time, urllib.request
from pathlib import Path
from typing import Any
import numpy as np
EPQS = "https://epqs.nationalmap.gov/v1/json?x={x}&y={y}&wkid=4326&units=Meters"
STEP = 10.0 / 111_111.0
SITES = {"flagstaff": (35.1983, -111.6513), "stonefield": (40.0150, -105.2705)}

def _elev(lon, lat):
    try:
        with urllib.request.urlopen(EPQS.format(x=lon, y=lat), timeout=12) as resp:
            return float(json.loads(resp.read().decode("utf-8"))["value"])
    except Exception:
        return None

def ingest(site="flagstaff", n=6):
    if site not in SITES:
        site = "flagstaff"
    lat0, lon0 = SITES[site]
    half = (n - 1) / 2.0
    zs = np.zeros((n, n)); pts = []; misses = 0
    for i in range(n):
        for j in range(n):
            lat = lat0 + (i - half) * STEP
            lon = lon0 + (j - half) * STEP
            z = _elev(lon, lat)
            if z is None:
                misses += 1; z = float("nan")
            zs[i, j] = z
            pts.append(((j - half) * 10.0, float(z) if z == z else 0.0, (i - half) * 10.0))
            time.sleep(0.05)
    finite = zs[np.isfinite(zs)]
    out_dir = Path(__file__).resolve().parent / "vault" / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / f"{site}.ply"
    lines = ["ply", "format ascii 1.0", f"element vertex {len(pts)}", "property float x", "property float y", "property float z", "end_header"]
    for x, y, z in pts:
        lines.append(f"{x:.3f} {y:.3f} {z:.3f}")
    ply.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row = {"site": site, "lat": lat0, "lon": lon0, "n": n, "step_m": 10.0, "source": "USGS EPQS",
           "resolution_note": "point samples; ~10 m, not 1 m LiDAR",
           "z_min": float(finite.min()) if finite.size else None,
           "z_max": float(finite.max()) if finite.size else None,
           "z_mean": float(finite.mean()) if finite.size else None,
           "misses": misses, "ply": str(ply), "bytes": ply.stat().st_size}
    (out_dir / f"{site}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row
