#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/backlight -> JuniorDeck sheet brightness.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md, Deck sheet package ports: the JuniorDeck
  sheet on 127.0.0.1:8766. The seam decides, from the backlight layer
  alone, how a Deck sheet brightness request (0..100 percent) may reach a
  panel: "apply" (write the mapped raw level), "clamp" (write a reduced
  level because of a power cap), or "hold" (leave the panel unchanged and
  report). SBC boards (JuniorOS T4) usually expose a raw PWM backlight;
  SFF hosts (JuniorHome T0) with an attached panel usually expose a
  firmware (ACPI) or platform node. Not ROADMAP T13 (overnight owns it).
  Not item 5 (ue5_launch stays False). Not a re-run of the USB HID, ALSA,
  or USB-C Deck seams: this is the display-backlight class only.

What it emulates
  In-memory fixtures shaped like /sys/class/backlight/<dev>/{type,
  brightness, max_brightness, actual_brightness, bl_power} plus a Deck
  sheet request {bind, percent, on_battery}. Rules, in order:
  1. Bind: must be exactly 127.0.0.1:8766; anything else rejects the row.
  2. Percent unparsed or outside 0..100 -> "hold".
  3. Device choice: prefer type firmware, then platform, then raw (the
     kernel's own preference order). No device -> "hold" (flag no_panel).
  4. Any chosen field unparsed, or max_brightness <= 0 -> "hold".
  5. bl_power != 0 (FB_BLANK_* not UNBLANK) -> "hold" (flag panel_blanked);
     the seam never unblanks a panel.
  6. actual_brightness differs from brightness by more than 5 percent of
     max -> flag readback_drift (still allowed; firmware may lag).
  7. Map percent to raw: raw devices use a perceptual square curve
     (raw = round(max * (p/100)^2)), firmware/platform use linear. A
     non-zero request never maps to raw 0 (minimum 1) so the sheet cannot
     black out the panel by rounding.
  8. Power cap: on_battery caps SBC at 60 percent and SFF at 80 percent
     of max -> "clamp" when the mapped level exceeds the cap.
  9. Otherwise -> "apply".
  Each row packs its decision and three check states (bind, device, power)
  as real 2-bit trit codes (0/1/2, four per byte, I2_S order: first code in
  the low bits), and rows are chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys reads or writes, no /dev/fb or DRM ioctl, no brightnessctl,
  no logind SetBrightness call, no sockets (binds are strings; nothing
  listens or connects), no device opened, no model pull, no GGUF, no
  network fetch, no subprocess. It does not import or re-run any team file
  or any earlier hwsw_* seam. Rows go to a tempfile removed after the run;
  nothing is written under ~/.juniorhome. Flags: mint, socket, layer_one,
  fetch, ue5_launch all False. JuniorSolana and the profile repo are
  untouched.

How to run
  python3 scripts/hwsw_backlight_deck_sheet_brightness_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_backlight_deck_sheet_brightness_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
BIND = "127.0.0.1:8766"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
TYPE_ORDER = ("firmware", "platform", "raw")
BATTERY_CAP = {"SBC": 0.60, "SFF": 0.80}
DRIFT_FRAC = 0.05
DECISION_CODE = {"hold": 0, "clamp": 1, "apply": 2}
CHECK_FAIL, CHECK_SKIP, CHECK_PASS = 0, 1, 2


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack_trits(codes):
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            if c not in (0, 1, 2):
                raise ValueError("trit code out of range")
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_trits(data, n):
    return [(data[k // 4] >> (2 * (k % 4))) & 3 for k in range(n)]


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def choose(devices):
    for t in TYPE_ORDER:
        for name in sorted(devices):
            if str(devices[name].get("type", "")).strip() == t:
                return name, t
    return None, None


def map_level(percent, mx, kind):
    if percent == 0:
        return 0
    frac = percent / 100.0
    raw = round(mx * frac * frac) if kind == "raw" else round(mx * frac)
    return max(1, min(mx, raw))


def decide(fx):
    """Return (decision, checks[bind, dev, power], flags, device, level)."""
    if fx.get("bind") != BIND:
        return "reject", [CHECK_FAIL, CHECK_SKIP, CHECK_SKIP], ["bad_bind"], None, None
    pct = _int(fx.get("percent"))
    if pct is None or not 0 <= pct <= 100:
        return "hold", [CHECK_PASS, CHECK_SKIP, CHECK_SKIP], ["percent_unparsed_or_range"], None, None
    name, kind = choose(fx.get("devices") or {})
    if name is None:
        return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["no_panel"], None, None
    d = fx["devices"][name]
    vals = {k: _int(d.get(k)) for k in ("brightness", "max_brightness", "actual_brightness", "bl_power")}
    if any(v is None for v in vals.values()) or vals["max_brightness"] <= 0:
        return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["dev_unparsed"], name, None
    if vals["bl_power"] != 0:
        return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["panel_blanked"], name, None
    flags = []
    mx = vals["max_brightness"]
    if abs(vals["actual_brightness"] - vals["brightness"]) > DRIFT_FRAC * mx:
        flags.append("readback_drift")
    level = map_level(pct, mx, kind)
    if fx.get("on_battery"):
        cap = max(1, int(mx * BATTERY_CAP[fx["cls"]]))
        if level > cap:
            return "clamp", [CHECK_PASS, CHECK_PASS, CHECK_FAIL], flags + ["battery_cap"], name, cap
    return "apply", [CHECK_PASS, CHECK_PASS, CHECK_PASS], flags, name, level


def _bl(kind, b="100", mx="255", act=None, pw="0"):
    return {"type": kind, "brightness": b, "max_brightness": mx,
            "actual_brightness": b if act is None else act, "bl_power": pw}


FIXTURES = [
    # name, fixture, expected decision, expected device, expected level
    ("sbc_pwm_raw_apply", {"cls": "SBC", "bind": BIND, "percent": "50",
                           "devices": {"backlight": _bl("raw")}}, "apply", "backlight", 64),
    ("sff_acpi_prefers_firmware", {"cls": "SFF", "bind": BIND, "percent": "50",
                                   "devices": {"intel_backlight": _bl("raw", mx="96000"),
                                               "acpi_video0": _bl("firmware", mx="100")}}, "apply", "acpi_video0", 50),
    ("sbc_battery_clamp", {"cls": "SBC", "bind": BIND, "percent": "100", "on_battery": True,
                           "devices": {"backlight": _bl("raw")}}, "clamp", "backlight", 153),
    ("sff_battery_under_cap_apply", {"cls": "SFF", "bind": BIND, "percent": "70", "on_battery": True,
                                     "devices": {"acpi_video0": _bl("firmware", mx="100")}}, "apply", "acpi_video0", 70),
    ("tiny_percent_never_zero", {"cls": "SBC", "bind": BIND, "percent": "1",
                                 "devices": {"backlight": _bl("raw")}}, "apply", "backlight", 1),
    ("panel_blanked_hold", {"cls": "SBC", "bind": BIND, "percent": "40",
                            "devices": {"backlight": _bl("raw", pw="4")}}, "hold", "backlight", None),
    ("no_panel_hold", {"cls": "SFF", "bind": BIND, "percent": "40", "devices": {}}, "hold", None, None),
    ("max_zero_hold", {"cls": "SBC", "bind": BIND, "percent": "40",
                       "devices": {"backlight": _bl("raw", mx="0")}}, "hold", "backlight", None),
    ("percent_range_hold", {"cls": "SBC", "bind": BIND, "percent": "140",
                            "devices": {"backlight": _bl("raw")}}, "hold", None, None),
    ("wildcard_bind_reject", {"cls": "SBC", "bind": "0.0.0.0:8766", "percent": "50",
                              "devices": {"backlight": _bl("raw")}}, "reject", None, None),
    ("drift_flag_apply", {"cls": "SFF", "bind": BIND, "percent": "30",
                          "devices": {"amdgpu_bl0": _bl("platform", b="200", mx="255", act="120")}}, "apply", "amdgpu_bl0", 76),
]


def run(verbose=False):
    prev = "0" * 16
    rows, failures = [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_backlight_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for name, fx, want, want_dev, want_level in FIXTURES:
                decision, checks, flags, dev, level = decide(fx)
                code = DECISION_CODE.get(decision, 0)
                codes = [code] + checks
                pack = pack_trits(codes)
                if unpack_trits(pack, len(codes)) != codes:
                    failures.append(name + ":pack_roundtrip")
                text = json.dumps({"row": name, "decision": decision, "device": dev, "level": level,
                                   "flags": flags, "rate": RATE}, sort_keys=True)
                c = close(prev, pack, text)
                fh.write(json.dumps({"close": c, "prev": prev, "pack": pack.hex(), "text": text}) + "\n")
                prev = c
                ok = decision == want and dev == want_dev and level == want_level
                if not ok:
                    failures.append("%s: got %s/%s/%s want %s/%s/%s" % (name, decision, dev, level, want, want_dev, want_level))
                rows.append({"row": name, "decision": decision, "device": dev, "level": level, "flags": flags, "close": c, "ok": ok})
                if verbose:
                    print("%-30s %-6s dev=%-16s level=%-6s flags=%s close=%s" % (name, decision, dev, level, ",".join(flags) or "-", c))
    finally:
        os.remove(path)
    if any(FLAGS.values()):
        failures.append("flags_not_false")
    return {"seam": "backlight->deck_sheet", "bind": BIND, "rate": RATE, "flags": FLAGS,
            "rows": len(rows), "passed": sum(r["ok"] for r in rows), "tip_close": prev,
            "failures": failures, "ok": not failures}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        res = run(verbose=True)
        print("tip_close=%s ok=%s" % (res["tip_close"], res["ok"]))
    else:
        res = run()
        print(json.dumps(res, indent=2))
    sys.exit(0 if res["ok"] else 1)
