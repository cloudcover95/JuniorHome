#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/drm EDID + connector -> JuniorDeck sheet mode.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md, Deck sheet package ports: the JuniorDeck
  sheet on 127.0.0.1:8766 (see also JuniorHome docs/DECK_HW.md, 10.1 in
  panel). The seam decides, from the DRM connector and EDID layer alone,
  which panel mode the Deck sheet may advertise: "apply" (use the chosen
  WxH@Hz), "clamp" (use a smaller mode that fits the Deck max), or "hold"
  (leave the sheet at its last known mode / report only). SBC boards
  (JuniorOS T4) usually expose a DSI or HDMI-A connector with a short
  EDID; SFF hosts (JuniorHome T0) with an attached panel expose
  eDP-1 / DP-1. Not ROADMAP T13 (overnight owns it). Not item 4 (no Asahi
  or MLX check; the DRM GPU Asahi probe seam is a different path). Not
  item 5 (ue5_launch stays False). Not a re-run of the backlight, USB-C,
  USB HID, ALSA, or DRM-GPU Asahi seams: this is connector status + EDID
  mode pick only.

What it emulates
  In-memory fixtures shaped like /sys/class/drm/<card>-<connector>/{status,
  enabled, modes, edid} plus a Deck sheet request {bind, class, max_w,
  max_h, max_hz}. "modes" is a newline list of "WxH" with the preferred
  mode marked by a trailing " *". "edid" is a hex string of the first
  128 bytes (or empty). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8766; anything else -> "hold"
     (flag bind_reject).
  2. Class must be "sbc" or "sff"; max_w/max_h/max_hz must parse and be
     positive. Otherwise -> "hold".
  3. status must be "connected", "disconnected", or "unknown". unknown
     or unparsed -> "hold".
  4. disconnected -> "clamp" to a soft virtual mode 800x480@60 (flag
     disconnected), regardless of EDID.
  5. EDID must start with the 8-byte header 00ffffffffffff00 when present;
     a non-empty EDID that fails the header -> "hold" (flag edid_bad).
     Empty EDID on a connected connector is allowed but flagged edid_missing.
  6. Parse modes as W x H; optional preferred marked with "*". Ignore
     blank lines. No parseable modes -> "hold" (flag no_modes).
  7. Drop modes with W>max_w or H>max_h or (when Hz is present) Hz>max_hz.
     If a preferred mode survives, pick it; else pick the largest area
     (W*H), ties prefer higher Hz then higher W.
  8. If the preferred/largest raw mode was dropped and a smaller one is
     used -> "clamp"; if the preferred (or sole largest) fits as-is ->
     "apply". If every mode was dropped -> "hold" (flag oversize).
  9. Hz above 120 on an applied/clamped pick is flagged high_refresh
     (still kept if <= max_hz).
  Each row packs its decision and three check states (bind, edid, mode)
  as real 2-bit trit codes (0/1/2, four per byte, I2_S order: first code
  in the low bits), and rows are chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys reads or writes, no /dev/dri access, no ioctl, no libdrm,
  no subprocess, no sockets (binds are strings; nothing listens or
  connects), no device opened, no model pull, no GGUF, no network fetch,
  no docker.sock. It does not import or re-run any team file or any
  earlier hwsw_* seam. Rows go to a tempfile removed after the run;
  nothing is written under ~/.juniorhome. Flags: mint, socket, layer_one,
  fetch, ue5_launch all False. JuniorSolana and the profile repo are
  untouched.

How to run
  python3 scripts/hwsw_drm_edid_deck_sheet_mode_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_drm_edid_deck_sheet_mode_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile

BIND = "127.0.0.1:8766"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"hold": 0, "clamp": 1, "apply": 2}
EDID_HEADER = "00ffffffffffff00"
SOFT_MODE = {"w": 800, "h": 480, "hz": 60}
MODE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)(?:\s*@\s*(\d+(?:\.\d+)?))?\s*(\*)?\s*$")


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
    codes = []
    for b in data:
        for j in range(4):
            codes.append((b >> (2 * j)) & 3)
    return codes[:n]


def to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_modes(text):
    out = []
    if not isinstance(text, str):
        return None
    for line in text.splitlines():
        if not line.strip():
            continue
        m = MODE_RE.match(line)
        if not m:
            return None
        w, h = int(m.group(1)), int(m.group(2))
        hz = float(m.group(3)) if m.group(3) else 60.0
        pref = bool(m.group(4))
        out.append({"w": w, "h": h, "hz": hz, "preferred": pref})
    return out


def edid_ok(edid):
    if edid is None:
        return False, "edid_unparsed"
    if not isinstance(edid, str):
        return False, "edid_unparsed"
    hexed = "".join(edid.split()).lower()
    if hexed == "":
        return True, "edid_missing"
    if len(hexed) < 16 or any(c not in "0123456789abcdef" for c in hexed):
        return False, "edid_bad"
    if not hexed.startswith(EDID_HEADER):
        return False, "edid_bad"
    return True, None


def pick_mode(modes, max_w, max_h, max_hz):
    fit = [m for m in modes if m["w"] <= max_w and m["h"] <= max_h and m["hz"] <= max_hz]
    if not fit:
        return None, True
    preferred = [m for m in fit if m.get("preferred")]
    pool = preferred or fit
    best = max(pool, key=lambda m: (m["w"] * m["h"], m["hz"], m["w"]))
    raw_best = max(modes, key=lambda m: (m["w"] * m["h"], m["hz"], m["w"]))
    clamped = not (raw_best["w"] <= max_w and raw_best["h"] <= max_h and raw_best["hz"] <= max_hz)
    if preferred and preferred[0] is best:
        clamped = False if (
            preferred[0]["w"] <= max_w and preferred[0]["h"] <= max_h and preferred[0]["hz"] <= max_hz
        ) else True
        # If we used preferred and it fits, apply; if preferred was dropped we wouldn't be here with preferred.
        clamped = False
    elif preferred:
        clamped = True
    else:
        # no preferred in fit: clamp if the absolute largest did not fit
        clamped = not (raw_best["w"] <= max_w and raw_best["h"] <= max_h and raw_best["hz"] <= max_hz)
    return {"w": best["w"], "h": best["h"], "hz": best["hz"]}, clamped


def decide(fx):
    """Return (decision, [bind, edid, mode] trit checks, flags, mode)."""
    flags = []
    req = fx.get("deck", {})
    if req.get("bind") != BIND:
        return "hold", [0, 0, 0], ["bind_reject"], None
    cls = req.get("class")
    max_w, max_h, max_hz = to_int(req.get("max_w")), to_int(req.get("max_h")), to_float(req.get("max_hz"))
    if cls not in ("sbc", "sff") or max_w is None or max_h is None or max_hz is None:
        return "hold", [2, 0, 0], ["bad_request"], None
    if max_w <= 0 or max_h <= 0 or max_hz <= 0:
        return "hold", [2, 0, 0], ["bad_request"], None

    conn = fx.get("connector", {})
    status = conn.get("status")
    if status not in ("connected", "disconnected", "unknown"):
        return "hold", [2, 0, 0], ["status_unparsed"], None
    if status == "unknown":
        return "hold", [2, 0, 0], ["status_unknown"], None
    if status == "disconnected":
        flags.append("disconnected")
        return "clamp", [2, 1, 2], flags, dict(SOFT_MODE)

    ok, eflag = edid_ok(conn.get("edid"))
    if not ok:
        return "hold", [2, 0, 0], [eflag], None
    if eflag:
        flags.append(eflag)
    edid_check = 1 if eflag == "edid_missing" else 2

    modes = parse_modes(conn.get("modes"))
    if modes is None:
        return "hold", [2, edid_check, 0], flags + ["modes_unparsed"], None
    if not modes:
        return "hold", [2, edid_check, 0], flags + ["no_modes"], None

    mode, clamped = pick_mode(modes, max_w, max_h, max_hz)
    if mode is None:
        return "hold", [2, edid_check, 0], flags + ["oversize"], None
    if mode["hz"] > 120:
        flags.append("high_refresh")
    decision = "clamp" if clamped else "apply"
    return decision, [2, edid_check, 2], flags, mode


# Minimal valid EDID header + padding zeros (128 hex chars = 64 bytes shown; enough for header check)
EDID_OK = EDID_HEADER + ("00" * 56)
EDID_BAD = "ffffffff" + ("00" * 60)

DECK_SBC = {"bind": BIND, "class": "sbc", "max_w": 1280, "max_h": 800, "max_hz": 60}
DECK_SFF = {"bind": BIND, "class": "sff", "max_w": 1920, "max_h": 1200, "max_hz": 60}


FIXTURES = [
    ("sbc_preferred_fits",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "1920x1080\n1280x800 *\n800x480\n",
                                      "edid": EDID_OK}},
     "apply", {"w": 1280, "h": 800, "hz": 60.0}),
    ("sbc_oversize_clamps",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "1920x1080 *\n1280x800\n1024x600\n",
                                      "edid": EDID_OK}},
     "clamp", {"w": 1280, "h": 800, "hz": 60.0}),
    ("sff_largest_apply",
     {"deck": DECK_SFF, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "1920x1200\n1600x900\n1280x800\n",
                                      "edid": EDID_OK}},
     "apply", {"w": 1920, "h": 1200, "hz": 60.0}),
    ("disconnected_soft",
     {"deck": DECK_SBC, "connector": {"status": "disconnected", "enabled": "disabled",
                                      "modes": "1920x1080 *\n", "edid": ""}},
     "clamp", dict(SOFT_MODE)),
    ("edid_missing_apply",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "1280x800 *\n", "edid": ""}},
     "apply", {"w": 1280, "h": 800, "hz": 60.0}),
    ("edid_bad_holds",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "1280x800 *\n", "edid": EDID_BAD}},
     "hold", None),
    ("no_modes_holds",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "\n", "edid": EDID_OK}},
     "hold", None),
    ("all_oversize_holds",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "3840x2160 *\n2560x1440\n", "edid": EDID_OK}},
     "hold", None),
    ("unknown_status_holds",
     {"deck": DECK_SBC, "connector": {"status": "unknown", "enabled": "enabled",
                                      "modes": "1280x800 *\n", "edid": EDID_OK}},
     "hold", None),
    ("bind_wildcard_rejects",
     {"deck": {"bind": "0.0.0.0:8766", "class": "sbc", "max_w": 1280, "max_h": 800, "max_hz": 60},
      "connector": {"status": "connected", "enabled": "enabled",
                    "modes": "1280x800 *\n", "edid": EDID_OK}},
     "hold", None),
    ("hz_present_pick",
     {"deck": {"bind": BIND, "class": "sff", "max_w": 1920, "max_h": 1200, "max_hz": 60},
      "connector": {"status": "connected", "enabled": "enabled",
                    "modes": "1920x1080@120\n1920x1080@60 *\n1280x720@60\n",
                    "edid": EDID_OK}},
     "apply", {"w": 1920, "h": 1080, "hz": 60.0}),
    ("modes_unparsed_holds",
     {"deck": DECK_SBC, "connector": {"status": "connected", "enabled": "enabled",
                                      "modes": "not-a-mode\n", "edid": EDID_OK}},
     "hold", None),
]


def run(verbose=False):
    prev = "0" * 16
    rows, failures = [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_edid_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for name, fx, want, want_mode in FIXTURES:
                decision, checks, flags, mode = decide(fx)
                codes = [DECISION_CODE[decision]] + checks
                pack = pack_trits(codes)
                if unpack_trits(pack, len(codes)) != codes:
                    failures.append(name + ":pack_roundtrip")
                # normalize mode hz to float for JSON stability
                mode_out = None
                if mode is not None:
                    mode_out = {"w": int(mode["w"]), "h": int(mode["h"]), "hz": float(mode["hz"])}
                text = json.dumps({"row": name, "decision": decision, "mode": mode_out,
                                   "flags": flags, "rate": RATE}, sort_keys=True)
                c = close(prev, pack, text)
                fh.write(json.dumps({"close": c, "prev": prev, "pack": pack.hex(), "text": text}) + "\n")
                prev = c
                want_norm = None
                if want_mode is not None:
                    want_norm = {"w": int(want_mode["w"]), "h": int(want_mode["h"]),
                                 "hz": float(want_mode["hz"])}
                ok = decision == want and mode_out == want_norm
                if name == "edid_missing_apply" and "edid_missing" not in flags:
                    ok = False
                if name == "disconnected_soft" and "disconnected" not in flags:
                    ok = False
                if name == "edid_bad_holds" and "edid_bad" not in flags:
                    ok = False
                if not ok:
                    failures.append("%s: got %s/%s want %s/%s flags=%s"
                                    % (name, decision, mode_out, want, want_norm, flags))
                rows.append({"row": name, "decision": decision, "mode": mode_out,
                             "flags": flags, "close": c, "ok": ok})
                if verbose:
                    print("%-24s %-5s mode=%-20s flags=%s close=%s"
                          % (name, decision, mode_out, ",".join(flags) or "-", c))
    finally:
        os.remove(path)
    if any(FLAGS.values()):
        failures.append("flags_not_false")
    return {"seam": "drm_edid->deck_sheet_mode", "bind": BIND, "rate": RATE, "flags": FLAGS,
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
