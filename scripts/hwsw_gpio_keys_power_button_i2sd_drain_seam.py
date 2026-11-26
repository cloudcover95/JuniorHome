#!/usr/bin/env python3
"""Hardware-to-software seam emulator: gpio-keys power button / lid switch -> i2sd graceful-drain plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where a GPIO power button or lid switch (as seen
  through /dev/input event-class sysfs attributes) tells i2sd to drain its
  receipt queue before shutdown. Hardware classes: SBC arm64 (JuniorOS T4,
  GPIO header button) and SFF host (JuniorHome T0, case power button / lid).
  Not T13, not item 2 (no GGUF), not item 4 (no Asahi/MLX), not item 5. No
  other team file or earlier hwsw_* seam is imported, read, or wrapped; GPIO
  button / lid switch input was not covered before.

What it emulates
  In-memory fixtures shaped like gpio-keys input attributes: key (KEY_POWER,
  KEY_SLEEP, SW_LID), state (0 released / 1 pressed or lid closed), held_ms,
  and wakeup (0/1). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Unknown key, state not 0/1, held_ms not a non-negative int, or wakeup
     not 0/1 -> hold (unparsed).
  4. state 0 -> admit (idle): no drain.
  5. KEY_POWER held >= 4000 ms -> soft (force_off): drain 0 receipts, mark spool dirty.
  6. KEY_POWER or KEY_SLEEP pressed -> soft (drain_30s): 30 s graceful drain.
  7. SW_LID closed on sff with wakeup 1 -> admit (lid_keep_alive), no drain;
     on sbc or wakeup 0 -> soft (drain_10s): 10 s drain.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys, /proc or /dev/input reads, no input device opened, no GPIO
  line requested, no systemd/logind, no shutdown call, no sockets (binds are
  strings), no network fetch, no subprocess, no model pull, no GGUF download.
  Rows live in memory only. Flags mint, socket, layer_one, fetch, ue5_launch
  all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_gpio_keys_power_button_i2sd_drain_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_gpio_keys_power_button_i2sd_drain_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
CLASSES = ("sbc", "sff")
KEYS = ("KEY_POWER", "KEY_SLEEP", "SW_LID")
FORCE_MS = 4000
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes):
    codes = list(codes)
    while len(codes) % 4:
        codes.append(1)
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            assert c in (0, 1, 2)
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def is01(v):
    return type(v) is int and v in (0, 1)


def decide(fx):
    """Return (decision, reason, drain_seconds, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASSES:
        return "hold", "bad_class", 0, [2, 2, 0]
    key, st, ms, wk = fx.get("key"), fx.get("state"), fx.get("held_ms"), fx.get("wakeup")
    if key not in KEYS or not is01(st) or not is01(wk) or type(ms) is not int or ms < 0:
        return "hold", "unparsed", 0, [2, 2, 2, 0]
    if st == 0:
        return "admit", "idle", 0, [2, 2, 2, 2]
    if key == "KEY_POWER" and ms >= FORCE_MS:
        return "soft", "force_off", 0, [2, 2, 2, 1]
    if key in ("KEY_POWER", "KEY_SLEEP"):
        return "soft", "drain_30s", 30, [2, 2, 2, 1]
    if fx["cls"] == "sff" and wk == 1:
        return "admit", "lid_keep_alive", 0, [2, 2, 2, 2]
    return "soft", "drain_10s", 10, [2, 2, 2, 1]


def K(key, st, ms=0, wk=0):
    return dict(key=key, state=st, held_ms=ms, wakeup=wk)


FIXTURES = [
    ("sbc_btn_idle", dict(cls="sbc", bind=BIND_OK, **K("KEY_POWER", 0)), "admit", 0),
    ("sbc_btn_tap", dict(cls="sbc", bind=BIND_OK, **K("KEY_POWER", 1, 250)), "soft", 30),
    ("sff_btn_long", dict(cls="sff", bind=BIND_OK, **K("KEY_POWER", 1, 4500)), "soft", 0),
    ("sff_sleep_key", dict(cls="sff", bind=BIND_OK, **K("KEY_SLEEP", 1, 100)), "soft", 30),
    ("sff_lid_wake", dict(cls="sff", bind=BIND_OK, **K("SW_LID", 1, 0, 1)), "admit", 0),
    ("sff_lid_nowake", dict(cls="sff", bind=BIND_OK, **K("SW_LID", 1, 0, 0)), "soft", 10),
    ("sbc_lid", dict(cls="sbc", bind=BIND_OK, **K("SW_LID", 1, 0, 1)), "soft", 10),
    ("bad_key", dict(cls="sff", bind=BIND_OK, **K("KEY_F13", 1)), "hold", 0),
    ("bad_state", dict(cls="sff", bind=BIND_OK, **K("KEY_POWER", 7)), "hold", 0),
    ("neg_held", dict(cls="sff", bind=BIND_OK, **K("KEY_POWER", 1, -5)), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, **K("KEY_POWER", 1)), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", **K("KEY_POWER", 1)), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, **K("KEY_POWER", 1)), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_s in FIXTURES:
        d, reason, s, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, d, reason, s))
        rows.append({"name": name, "decision": d, "reason": reason, "drain_s": s, "pack": pk.hex(), "close": prev})
        if (d, s) != (want_d, want_s):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-16s %-6s %-14s drain=%ds pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["drain_s"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
