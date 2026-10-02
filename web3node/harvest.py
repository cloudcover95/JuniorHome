"""Harvest class critique. Bands are published, not measured here."""
from __future__ import annotations

import json
from pathlib import Path

from trit_mesh import integrity

OUT = Path.home() / ".juniorhome" / "os" / "harvest.json"
# yield and cost are ranks, 1 = low. Not dollars.
BANDS = {
    "solar": {"yield": 3, "cost": 2, "note": "outdoor mW/cm2 class, indoor uW"},
    "thermal": {"yield": 2, "cost": 3, "note": "uW/cm3 at low dT"},
    "kinetic": {"yield": 2, "cost": 2, "note": "vibration, duty cycle"},
    "rf": {"yield": 1, "cost": 2, "note": "ambient nW-uW"},
}


def critique(note: str = "harvest") -> dict:
    energy = integrity(note)["energy"]
    rows = []
    for name, band in BANDS.items():
        score = round(band["yield"] / band["cost"], 3)
        rows.append({"name": name, "score": score, **band})
    rows.sort(key=lambda r: r["score"], reverse=True)
    body = {
        "protocol": "goldend-osai-omega/1",
        "trit_energy": energy,
        "measured": False,
        "rows": rows,
        "best": rows[0]["name"],
        "train": False,
        "bind": "127.0.0.1",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
