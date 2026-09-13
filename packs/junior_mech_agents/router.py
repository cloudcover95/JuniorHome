"""Ternary mech-agent router. Stdlib. 45 W node.

Maps a 16-float feature window (scan / CAD / bar ticks) to one of
six agent actions via AbsMean ternary matvec. Same math as FFBN.

Actions never write FrameForge knockback. Stock path is intent only.
SolidWorks is an interchange target (STEP/STL), not a COM host.
"""
from __future__ import annotations

ACTIONS = (
    "snap_scan",
    "emit_glb",
    "export_step",
    "fea_flag",
    "stock_intent",
    "halt",
)


def absmean(xs):
    if not xs:
        return 0.0
    return sum(abs(x) for x in xs) / len(xs)


def quant_vec(xs, keep=0.55):
    scale = absmean(xs)
    if scale <= 1e-12:
        return [0] * len(xs), 0.0
    scored = sorted(((abs(x), i) for i, x in enumerate(xs)), reverse=True)
    keep_n = max(1, int(len(xs) * keep))
    live = {i for _, i in scored[:keep_n]}
    out = []
    for i, x in enumerate(xs):
        if i not in live:
            out.append(0)
            continue
        v = x / scale
        out.append(1 if v > 0.5 else (-1 if v < -0.5 else 0))
    return out, scale


def matvec(w, rows, cols, x, scale):
    out = [0.0] * rows
    for r in range(rows):
        acc = 0.0
        base = r * cols
        for c in range(cols):
            t = w[base + c]
            if t:
                acc += x[c] if t > 0 else -x[c]
        out[r] = acc * scale
    return out


def pack_features(scan_density, svd_stability, mesh_watts, cad_dirty, fea_stress, bar_mom, bar_vol, halt=0.0):
    return [scan_density, svd_stability, mesh_watts, cad_dirty, fea_stress, bar_mom, bar_vol, halt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _wired_weights(rows, cols):
    w = [0] * (rows * cols)
    for row, cols_i in ((0, (0, 1)), (1, (2,)), (2, (3,)), (3, (4,)), (4, (5, 6)), (5, (7,))):
        for c in cols_i:
            w[row * cols + c] = 1
    return w, 1.0


def route(feat, seed=None):
    rows, cols = 6, 16
    x = (list(feat) + [0.0] * cols)[:cols]
    if seed is not None:
        w, scale = quant_vec((list(seed) + [0.0] * (rows * cols))[: rows * cols])
    else:
        w, scale = _wired_weights(rows, cols)
    logits = matvec(w, rows, cols, x, scale)
    idx = max(range(len(logits)), key=lambda i: logits[i])
    return {
        "action": ACTIONS[idx],
        "index": idx,
        "logits": [round(v, 5) for v in logits],
        "nonzero": sum(1 for t in w if t),
        "scale": round(scale, 4),
        "envelope": {
            "port": "JuniorBitNetFieldCore",
            "tree": {
                "snap_scan": "JuniorOmega.sensors",
                "emit_glb": "JuniorOmega.blender",
                "export_step": "JuniorOmega.cad",
                "fea_flag": "JuniorEngrTools.fea",
                "stock_intent": "JuniorStock",
                "halt": "JuniorHome",
            }.get(ACTIONS[idx], "JuniorHome"),
        },
    }


def self_test():
    dense = route(pack_features(0.9, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0))
    dirty = route(pack_features(0.1, 0.8, 0.2, 0.95, 0.1, 0.0, 0.0))
    bar = route(pack_features(0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8))
    assert dense["action"] == "snap_scan", dense
    assert dirty["action"] == "export_step", dirty
    assert bar["action"] == "stock_intent", bar
    return {"ok": True, "dense": dense["action"], "dirty": dirty["action"], "bar": bar["action"]}
