"""BitnetCloud sidecar surface.

Does not rewrite FrameForge or FrameForge2D kernels.
BitNet scores CPU intent only. Knockback stays in math_kb.py.
Not a Nintendo product.
"""
from __future__ import annotations

from . import host


def intent_to_sim(machine):
    snap = machine.snapshot()
    return {
        "kind": "cpu_intent",
        "pad": snap["pad"],
        "trit": snap["intent"]["trit"],
        "label": snap["intent"]["label"],
        "format": snap.get("format"),
        "xr": snap.get("xr"),
        "scale_knockback": False,
        "sidecar": "bitnetCloud",
        "legal": "not a nintendo product",
    }


def from_crispy(packet, profile="generic_iot"):
    m = host.Machine(profile=profile)
    m.ingest_crispy(packet)
    m.xr_beta({"source": "crispy-mouse"})
    return intent_to_sim(m)
