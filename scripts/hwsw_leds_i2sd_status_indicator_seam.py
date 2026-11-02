#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/leds -> i2sd status indicator.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767.
  No docker.sock." The seam decides, from the LED class layer alone, how
  the i2sd userspace daemon's state (serving / degraded / held / stopped)
  may be shown on a board status LED: "apply" (the overlay may set the
  LED as computed), "soft" (report the state through juniorctl text only,
  leave every LED untouched), or "hold" (the row is unusable; nothing is
  shown). SBC boards (JuniorOS T4) usually expose a devicetree status or
  heartbeat LED; SFF hosts (JuniorHome T0) often expose only keyboard or
  disk LEDs, which the seam never borrows. Not ROADMAP T13 (overnight
  owns it). Not item 4 (no Asahi or MLX check here). Not item 5
  (ue5_launch stays False). Not a re-run of the backlight, watchdog,
  thermal, power, or any other earlier hwsw seam: this is the LED class.

What it emulates
  In-memory fixtures shaped like /sys/class/leds/<devicename:color:function>/
  {brightness, max_brightness, trigger} (trigger lists the available
  triggers with the active one in [brackets]) plus an i2sd report
  {class, bind, state}. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767; anything else rejects the row
     (decision "hold", flag bind_reject).
  2. Class must be "sbc" or "sff"; state must be one of serving, degraded,
     held, stopped. Otherwise -> "hold".
  3. LED choice: names split as devicename:color:function. Functions in
     FOREIGN (disk-activity, mmc, wlan, lan, bluetooth, capslock, numlock,
     scrolllock, kbd_backlight, charging, flash, torch, mute, micmute) are
     never used. Prefer function "status", then "indicator", then
     "heartbeat". None left -> "soft" (flag no_led).
  4. max_brightness unparsed or <= 0, or trigger unparsed (no bracketed
     active entry) -> "hold".
  5. Active trigger other than "none" or "timer" means the kernel or
     another daemon owns the LED -> "soft" (flag trigger_busy); the seam
     never overrides it.
  6. Pattern: serving -> steady at max; degraded -> steady at max//2
     (minimum 1); held -> "timer" trigger 500/500 ms when "timer" is
     available, else steady at 1 (flag no_timer); stopped -> steady 0.
  7. A red LED showing "serving" is flagged color_mismatch (still applied).
  8. Otherwise -> "apply".
  Each row packs its decision and three check states (bind, led, trigger)
  as real 2-bit trit codes (0/1/2, four per byte, I2_S order: first code
  in the low bits), and rows are chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys reads or writes, no /dev access, no ioctl, no subprocess,
  no sockets (binds are strings; nothing listens or connects), no device
  opened, no model pull, no GGUF, no network fetch, no docker.sock. It
  does not import or re-run any team file or any earlier hwsw_* seam.
  Rows go to a tempfile removed after the run; nothing is written under
  ~/.juniorhome. Flags: mint, socket, layer_one, fetch, ue5_launch all
  False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_leds_i2sd_status_indicator_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_leds_i2sd_status_indicator_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile

BIND = "127.0.0.1:8767"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"hold": 0, "soft": 1, "apply": 2}
STATES = ("serving", "degraded", "held", "stopped")
FOREIGN = {"disk-activity", "mmc", "wlan", "lan", "bluetooth", "capslock", "numlock",
           "scrolllock", "kbd_backlight", "charging", "flash", "torch", "mute", "micmute"}
PREFER = ("status", "indicator", "heartbeat")
TIMER_MS = (500, 500)


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


def split_name(name):
    parts = name.split(":")
    while len(parts) < 3:
        parts.insert(0, "")
    return parts[-3], parts[-2], parts[-1]


def parse_trigger(text):
    if not isinstance(text, str):
        return None, []
    avail = text.split()
    active = [t[1:-1] for t in avail if re.fullmatch(r"\[[^\[\]\s]+\]", t)]
    names = [t.strip("[]") for t in avail]
    if len(active) != 1:
        return None, names
    return active[0], names


def to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def choose_led(leds):
    ranked = []
    for name in sorted(leds):
        _dev, _color, func = split_name(name)
        if func in FOREIGN or func not in PREFER:
            continue
        ranked.append((PREFER.index(func), name))
    return min(ranked)[1] if ranked else None


def decide(fx):
    """Return (decision, [bind, led, trigger] trit checks, flags, led, pattern)."""
    flags = []
    rep = fx.get("i2sd", {})
    if rep.get("bind") != BIND:
        return "hold", [0, 0, 0], ["bind_reject"], None, None
    if rep.get("class") not in ("sbc", "sff") or rep.get("state") not in STATES:
        return "hold", [2, 0, 0], ["bad_report"], None, None
    led = choose_led(fx.get("leds", {}))
    if led is None:
        return "soft", [2, 1, 1], ["no_led"], None, None
    node = fx["leds"][led]
    mx = to_int(node.get("max_brightness"))
    active, avail = parse_trigger(node.get("trigger"))
    if mx is None or mx <= 0 or active is None:
        return "hold", [2, 0, 0], ["led_unparsed"], led, None
    if active not in ("none", "timer"):
        return "soft", [2, 2, 1], ["trigger_busy:" + active], led, None
    state = rep["state"]
    if state == "serving":
        pattern = {"trigger": "none", "brightness": mx}
    elif state == "degraded":
        pattern = {"trigger": "none", "brightness": max(1, mx // 2)}
    elif state == "held":
        if "timer" in avail:
            pattern = {"trigger": "timer", "brightness": mx,
                       "delay_on": TIMER_MS[0], "delay_off": TIMER_MS[1]}
        else:
            pattern = {"trigger": "none", "brightness": 1}
            flags.append("no_timer")
    else:
        pattern = {"trigger": "none", "brightness": 0}
    if split_name(led)[1] == "red" and state == "serving":
        flags.append("color_mismatch")
    return "apply", [2, 2, 2], flags, led, pattern


TRIG = "none kbd-capslock disk-activity timer heartbeat mmc0"


def trig(active):
    return " ".join("[%s]" % t if t == active else t for t in TRIG.split())


SBC_OK = {"i2sd": {"class": "sbc", "bind": BIND, "state": "serving"}}

FIXTURES = [
    ("sbc_status_serving", dict(SBC_OK, leds={"board:green:status": {"brightness": "0", "max_brightness": "1", "trigger": trig("none")},
                                              "board:green:disk-activity": {"brightness": "0", "max_brightness": "1", "trigger": trig("disk-activity")}}),
     "apply", "board:green:status", {"trigger": "none", "brightness": 1}),
    ("sbc_degraded_half", {"i2sd": {"class": "sbc", "bind": BIND, "state": "degraded"},
                           "leds": {"pwr:white:status": {"brightness": "255", "max_brightness": "255", "trigger": trig("none")}}},
     "apply", "pwr:white:status", {"trigger": "none", "brightness": 127}),
    ("sbc_held_timer", {"i2sd": {"class": "sbc", "bind": BIND, "state": "held"},
                        "leds": {"act:green:indicator": {"brightness": "0", "max_brightness": "1", "trigger": trig("timer")}}},
     "apply", "act:green:indicator", {"trigger": "timer", "brightness": 1, "delay_on": 500, "delay_off": 500}),
    ("sbc_held_no_timer", {"i2sd": {"class": "sbc", "bind": BIND, "state": "held"},
                           "leds": {"act:green:status": {"brightness": "0", "max_brightness": "8", "trigger": "[none] heartbeat"}}},
     "apply", "act:green:status", {"trigger": "none", "brightness": 1}),
    ("sbc_heartbeat_busy", {"i2sd": {"class": "sbc", "bind": BIND, "state": "serving"},
                            "leds": {"act:green:heartbeat": {"brightness": "1", "max_brightness": "1", "trigger": trig("heartbeat")}}},
     "soft", "act:green:heartbeat", None),
    ("sff_only_kbd_disk", {"i2sd": {"class": "sff", "bind": BIND, "state": "serving"},
                           "leds": {"input3::capslock": {"brightness": "0", "max_brightness": "1", "trigger": trig("kbd-capslock")},
                                    "input3::kbd_backlight": {"brightness": "0", "max_brightness": "3", "trigger": trig("none")}}},
     "soft", None, None),
    ("sff_status_stopped", {"i2sd": {"class": "sff", "bind": BIND, "state": "stopped"},
                            "leds": {"chassis:amber:status": {"brightness": "1", "max_brightness": "1", "trigger": trig("none")}}},
     "apply", "chassis:amber:status", {"trigger": "none", "brightness": 0}),
    ("red_serving_flagged", {"i2sd": {"class": "sbc", "bind": BIND, "state": "serving"},
                             "leds": {"sys:red:status": {"brightness": "0", "max_brightness": "1", "trigger": trig("none")}}},
     "apply", "sys:red:status", {"trigger": "none", "brightness": 1}),
    ("max_zero_holds", {"i2sd": {"class": "sbc", "bind": BIND, "state": "serving"},
                        "leds": {"sys:green:status": {"brightness": "0", "max_brightness": "0", "trigger": trig("none")}}},
     "hold", "sys:green:status", None),
    ("trigger_unparsed_holds", {"i2sd": {"class": "sbc", "bind": BIND, "state": "serving"},
                                "leds": {"sys:green:status": {"brightness": "0", "max_brightness": "1", "trigger": "none timer"}}},
     "hold", "sys:green:status", None),
    ("bind_wildcard_rejects", {"i2sd": {"class": "sbc", "bind": "0.0.0.0:8767", "state": "serving"},
                               "leds": {"sys:green:status": {"brightness": "0", "max_brightness": "1", "trigger": trig("none")}}},
     "hold", None, None),
    ("unknown_state_holds", {"i2sd": {"class": "sbc", "bind": BIND, "state": "booting"},
                             "leds": {"sys:green:status": {"brightness": "0", "max_brightness": "1", "trigger": trig("none")}}},
     "hold", None, None),
]


def run(verbose=False):
    prev = "0" * 16
    rows, failures = [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_leds_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for name, fx, want, want_led, want_pat in FIXTURES:
                decision, checks, flags, led, pattern = decide(fx)
                codes = [DECISION_CODE[decision]] + checks
                pack = pack_trits(codes)
                if unpack_trits(pack, len(codes)) != codes:
                    failures.append(name + ":pack_roundtrip")
                text = json.dumps({"row": name, "decision": decision, "led": led, "pattern": pattern,
                                   "flags": flags, "rate": RATE}, sort_keys=True)
                c = close(prev, pack, text)
                fh.write(json.dumps({"close": c, "prev": prev, "pack": pack.hex(), "text": text}) + "\n")
                prev = c
                ok = decision == want and led == want_led and pattern == want_pat
                if name == "red_serving_flagged" and "color_mismatch" not in flags:
                    ok = False
                if not ok:
                    failures.append("%s: got %s/%s/%s want %s/%s/%s" % (name, decision, led, pattern, want, want_led, want_pat))
                rows.append({"row": name, "decision": decision, "led": led, "pattern": pattern,
                             "flags": flags, "close": c, "ok": ok})
                if verbose:
                    print("%-24s %-5s led=%-22s pattern=%s flags=%s close=%s"
                          % (name, decision, led, pattern, ",".join(flags) or "-", c))
    finally:
        os.remove(path)
    if any(FLAGS.values()):
        failures.append("flags_not_false")
    return {"seam": "leds->i2sd_status", "bind": BIND, "rate": RATE, "flags": FLAGS,
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
