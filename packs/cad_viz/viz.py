"""IoT-edge CAD / Blender visualizer stub. Stdlib SVG AABB."""
from __future__ import annotations


def aabb(points):
    if not points:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0], "span": 0.0}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    mn = [min(xs), min(ys), min(zs)]
    mx = [max(xs), max(ys), max(zs)]
    span = max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-6)
    return {"min": mn, "max": mx, "span": span}


def lod_trit(n):
    if n < 200:
        return -1
    if n < 2000:
        return 0
    return 1


def svg_box(box, w=240, h=160):
    span = max(box["span"], 1e-6)
    bw = (box["max"][0] - box["min"][0]) / span * (w - 24)
    bh = (box["max"][1] - box["min"][1]) / span * (h - 24)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<rect x="8" y="8" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="#3d5c3a" stroke-width="2"/>'
        f'<text x="12" y="{h - 8}" font-size="10" fill="#3d5c3a">span {span:.3f}</text></svg>'
    )


def preview(points):
    box = aabb(points)
    return {"n": len(points), "lod": lod_trit(len(points)), "aabb": box, "svg": svg_box(box), "next": "JuniorOmega.blender" if points else "halt"}


def self_test():
    p = preview([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
    assert p["n"] == 4 and p["lod"] == -1 and "<svg" in p["svg"]
    return {"ok": True, "lod": p["lod"], "span": p["aabb"]["span"]}
