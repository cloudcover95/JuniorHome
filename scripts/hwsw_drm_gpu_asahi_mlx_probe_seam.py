#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/drm GPU nodes -> Asahi MLX probe verdict.

Roadmap item
  JuniorHome docs/ROADMAP.md item 4, "Asahi MLX -- probe only from
  automations." This seam reads only the GPU layer (DRM card and render
  nodes plus the devicetree compatible string) and reports whether an Apple
  AGX GPU under the Asahi driver is *seen*. It never admits MLX: mlx_admit is
  always False and the backend stays numpy / CPU i2sd. Not ROADMAP T13
  (overnight owns it). Not item 5: a discrete GPU never flips ue5_launch,
  which stays False whatever the card is.

What it emulates
  In-memory fixtures shaped like /sys/class/drm/<cardN|renderDN>/device/
  uevent (DRIVER=...), vendor / device ids for PCI GPUs, plus
  /proc/device-tree/compatible for arm64 boxes. Hardware classes: SBC
  (arm64, JuniorOS T4), SFF host (x86_64 or arm64, JuniorHome T0), and Apple
  silicon under Asahi (probe only).
  1. Driver pick per card: "asahi" -> agx; "i915"/"xe" -> intel;
     "amdgpu"/"radeon" -> amd; "nouveau"/"nvidia" -> nvidia; "simpledrm",
     "efidrm", "vc4", "v3d", "panfrost", "lima", "rockchip" -> soc_or_fb;
     anything else or a missing DRIVER line -> unknown.
  2. mlx_seen is True only when a card uses the asahi driver AND the
     devicetree compatible names an "apple,t8xxx" / "apple,t6xxx" SoC AND a
     renderD node exists. A mismatch (asahi driver without apple compatible,
     or apple compatible without render node) holds as "probe_mismatch".
  3. mlx_admit is always False; backend is always "numpy"; i2sd bind must be
     exactly the string 127.0.0.1:8767 or the row rejects. A 0.0.0.0 bind
     rejects. The render node path is reported, never opened.
  4. Class check: an SBC fixture with a discrete PCI GPU (intel/amd/nvidia)
     is flagged "class_mismatch" (SBC is <=12 W, no dGPU); SFF accepts any.
  5. Correctness guard: a fixed ternary block dot (2-bit I2_S codes) on the
     numpy-equivalent pure-Python path equals the scalar reference, so the
     CPU fallback the probe keeps is exercised every row.
  One row per fixture is chained with close rule sha256(prev || pack || text)[:16]
  at rate 0.05, where pack is real 2-bit trit codes (0/1/2, four per byte,
  I2_S order) built from driver, compatible, render, class, and bind states.

What it does not touch
  No real /sys or /proc reads, no /dev/dri open, no ioctl, no Mesa, no
  Metal, no MLX import, no sockets (binds are strings; nothing listens or
  connects), no model pull, no GGUF, no network fetch, no subprocess. It
  does not import or re-run any team file (advance_asahi_mlx_* included) or
  any earlier hwsw_* seam (SFF T0 cpuinfo Asahi fixture, CPU ISA, thermal,
  memory, storage, power, USB-C, ALSA, and so on). Rows go to a tempfile
  removed after the run; nothing under ~/.juniorhome.
  Flags: mint, socket, layer_one, fetch, ue5_launch all False.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_drm_gpu_asahi_mlx_probe_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_drm_gpu_asahi_mlx_probe_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DRIVER_KIND = {
    "asahi": "agx", "i915": "intel", "xe": "intel", "amdgpu": "amd", "radeon": "amd",
    "nouveau": "nvidia", "nvidia": "nvidia", "simpledrm": "soc_or_fb", "efidrm": "soc_or_fb",
    "vc4": "soc_or_fb", "v3d": "soc_or_fb", "panfrost": "soc_or_fb", "lima": "soc_or_fb",
    "rockchip": "soc_or_fb",
}
DISCRETE = {"intel", "amd", "nvidia"}
APPLE_SOC_PREFIX = ("apple,t8", "apple,t6")

W_CODES = [(i * 7 + 2) % 3 for i in range(64)]
ACTS = [((i * 37) % 251) - 125 for i in range(64)]


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes: list) -> bytes:
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            b |= (int(c) & 3) << (2 * j)
        out.append(b)
    return bytes(out)


def unpack2(data: bytes, n: int) -> list:
    return [(data[k // 4] >> (2 * (k % 4))) & 3 for k in range(n)]


def dot_scalar(codes: list, acts: list) -> int:
    return sum((c - 1) * a for c, a in zip(codes, acts))


def dot_packed_blocks(packed: bytes, acts: list, block: int = 16) -> int:
    codes = unpack2(packed, len(acts))
    return sum(dot_scalar(codes[i:i + block], acts[i:i + block]) for i in range(0, len(acts), block))


def driver_of(uevent: str) -> str:
    for line in (uevent or "").splitlines():
        if line.startswith("DRIVER="):
            return line.split("=", 1)[1].strip()
    return ""


def evaluate(fx: dict) -> dict:
    cards = fx.get("cards", {})
    kinds = []
    render = None
    for name in sorted(cards):
        kind = DRIVER_KIND.get(driver_of(cards[name].get("uevent", "")), "unknown")
        if name.startswith("renderD"):
            render = render or ("/dev/dri/" + name)
        kinds.append(kind)
    compat = [c for c in fx.get("compatible", "").split("\0") if c]
    apple = any(c.startswith(APPLE_SOC_PREFIX) for c in compat)
    agx = "agx" in kinds
    verdict, reason = "ok", ""
    if fx.get("bind") != BIND_OK:
        verdict, reason = "reject", "bind_not_loopback_8767"
    elif agx and apple and render:
        verdict, reason = "probe_seen", "agx+apple+render"
    elif agx or apple:
        verdict, reason = "hold", "probe_mismatch"
    if verdict != "reject" and fx["cls"] == "sbc" and DISCRETE.intersection(kinds):
        verdict, reason = "hold", "class_mismatch"
    packed = pack2(W_CODES)
    dot_ok = dot_packed_blocks(packed, ACTS) == dot_scalar(W_CODES, ACTS)
    trits = [
        2 if agx else (0 if "unknown" in kinds or not kinds else 1),
        2 if apple else 1,
        2 if render else 0,
        0 if reason == "class_mismatch" else 1,
        2 if fx.get("bind") == BIND_OK else 0,
        2 if dot_ok else 0,
    ]
    return {"name": fx["name"], "cls": fx["cls"], "kinds": kinds, "render_node": render,
            "mlx_seen": verdict == "probe_seen", "mlx_admit": False, "backend": "numpy",
            "verdict": verdict, "reason": reason, "dot_ok": dot_ok, "trits": trits}


FIXTURES = [
    {"name": "asahi_m1_mini", "cls": "sff", "bind": BIND_OK, "compatible": "apple,j274\0apple,t8103\0apple,arm-platform",
     "cards": {"card0": {"uevent": "DRIVER=asahi"}, "renderD128": {"uevent": "DRIVER=asahi"}},
     "want": ("probe_seen", True)},
    {"name": "asahi_no_render", "cls": "sff", "bind": BIND_OK, "compatible": "apple,t6000",
     "cards": {"card0": {"uevent": "DRIVER=asahi"}}, "want": ("hold", False)},
    {"name": "apple_fb_only", "cls": "sff", "bind": BIND_OK, "compatible": "apple,t8112",
     "cards": {"card0": {"uevent": "DRIVER=simpledrm"}}, "want": ("hold", False)},
    {"name": "sff_x86_intel", "cls": "sff", "bind": BIND_OK, "compatible": "",
     "cards": {"card0": {"uevent": "DRIVER=i915"}, "renderD128": {"uevent": "DRIVER=i915"}},
     "want": ("ok", False)},
    {"name": "sbc_v3d", "cls": "sbc", "bind": BIND_OK, "compatible": "raspberrypi,5-model-b\0brcm,bcm2712",
     "cards": {"card0": {"uevent": "DRIVER=v3d"}, "renderD128": {"uevent": "DRIVER=v3d"}},
     "want": ("ok", False)},
    {"name": "sbc_with_dgpu", "cls": "sbc", "bind": BIND_OK, "compatible": "rockchip,rk3588",
     "cards": {"card0": {"uevent": "DRIVER=amdgpu"}}, "want": ("hold", False)},
    {"name": "asahi_open_bind", "cls": "sff", "bind": "0.0.0.0:8767", "compatible": "apple,t8103",
     "cards": {"card0": {"uevent": "DRIVER=asahi"}, "renderD128": {"uevent": "DRIVER=asahi"}},
     "want": ("reject", False)},
]


def run(demo: bool = False) -> dict:
    prev, fails, rows = GENESIS, [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_drm_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for fx in FIXTURES:
                r = evaluate(fx)
                text = json.dumps({k: r[k] for k in ("name", "verdict", "reason", "mlx_seen")}, sort_keys=True)
                prev = close(prev, pack2(r["trits"]), text)
                r["close"] = prev
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                ok = (r["verdict"], r["mlx_seen"]) == fx["want"] and r["mlx_admit"] is False and r["dot_ok"]
                if not ok:
                    fails.append(fx["name"])
                rows.append(r)
                if demo:
                    print(f"{r['name']:<18} {r['cls']} verdict={r['verdict']:<10} reason={r['reason'] or '-':<22} "
                          f"seen={r['mlx_seen']} admit=False render={r['render_node']} close={prev}")
    finally:
        os.remove(path)
    if any(FLAGS.values()):
        fails.append("flags")
    return {"seam": "drm_gpu->asahi_mlx_probe", "roadmap": "ROADMAP #4 Asahi MLX probe only", "rate": RATE,
            "rows": len(rows), "flags": FLAGS, "fails": fails, "pass": not fails, "tip_close": prev}


if __name__ == "__main__":
    out = run(demo=len(sys.argv) > 1 and sys.argv[1] == "demo")
    print(json.dumps(out, indent=2, sort_keys=True))
    sys.exit(0 if out["pass"] else 1)
