#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/rfkill radio switches -> i2sd loopback-only admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where rfkill switch state (wlan, bluetooth,
  wwan) decides whether i2sd runs as plain loopback service or in a
  loopback-only "quiet" mode. Hardware classes: SBC arm64 (JuniorOS T4)
  and SFF host (JuniorHome T0). Not T13, not item 2 (no GGUF), not item 4
  (no Asahi/MLX), not item 5. No other team file or earlier hwsw_* seam is
  imported, read, or wrapped; no rfkill file was covered before.

What it emulates
  In-memory fixtures shaped like rfkill sysfs attributes: type, soft (0/1),
  hard (0/1) as the kernel writes them. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Any soft/hard value not 0/1 -> hold (unparsed).
  4. Hard-blocked radio (hard=1) -> soft decision (hard_block): hardware
     switch wins, i2sd stays loopback and sheds half its threads.
  5. All declared radios soft-blocked (airplane mode) -> admit (quiet):
     loopback needs no radio.
  6. SBC with no radios declared -> soft (no_radio_info); SFF may have none.
  7. Otherwise admit.
  Threads: SBC 4, SFF 8, capped by wanted; soft halves (floor, min 1);
  hold/reject gives 0. Rows pack decision and check states as 2-bit trit
  codes (four per byte, I2_S order, first code in low bits) and chain with
  close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no rfkill writes, no /dev/rfkill, no rfkill
  binary, no systemd, no sockets (binds are strings), no device opened, no
  network fetch, no subprocess, no model pull, no GGUF download. Rows live
  in memory only. Flags mint, socket, layer_one, fetch, ue5_launch all
  False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_rfkill_radio_i2sd_loopback_only_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_rfkill_radio_i2sd_loopback_only_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
CLASS_THREADS = {"sbc": 4, "sff": 8}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}


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


def decide(fx):
    """Return (decision, reason, threads, codes)."""
    def done(d, r, codes):
        t = 0
        if d in ("admit", "soft"):
            t = min(CLASS_THREADS[fx["cls"]], fx.get("want", 99))
            if d == "soft":
                t = max(1, t // 2)
        return d, r, t, [DECISION_CODE[d]] + codes

    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASS_THREADS:
        return "hold", "bad_class", 0, [2, 2, 0]
    radios = fx.get("radios", [])
    for r in radios:
        if r.get("soft") not in (0, 1) or r.get("hard") not in (0, 1):
            return done("hold", "unparsed", [2, 2, 0])
    if any(r["hard"] == 1 for r in radios):
        return done("soft", "hard_block", [2, 2, 1])
    if radios and all(r["soft"] == 1 for r in radios):
        return done("admit", "quiet", [2, 2, 2])
    if not radios and fx["cls"] == "sbc":
        return done("soft", "no_radio_info", [2, 2, 1])
    return done("admit", "ok", [2, 2, 2])


R = lambda t, s, h: {"type": t, "soft": s, "hard": h}
FIXTURES = [
    ("sbc_wlan_up", dict(cls="sbc", bind=BIND_OK, want=8, radios=[R("wlan", 0, 0)]), "admit", 4),
    ("sff_no_radios", dict(cls="sff", bind=BIND_OK, want=8, radios=[]), "admit", 8),
    ("sbc_no_radios", dict(cls="sbc", bind=BIND_OK, want=8, radios=[]), "soft", 2),
    ("sbc_hard_block", dict(cls="sbc", bind=BIND_OK, want=8, radios=[R("wlan", 0, 1), R("bluetooth", 0, 0)]), "soft", 2),
    ("sff_airplane", dict(cls="sff", bind=BIND_OK, want=3, radios=[R("wlan", 1, 0), R("bluetooth", 1, 0)]), "admit", 3),
    ("sff_hard_one_thread", dict(cls="sff", bind=BIND_OK, want=1, radios=[R("wwan", 0, 1)]), "soft", 1),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, radios=[]), "hold", 0),
    ("unparsed_soft", dict(cls="sff", bind=BIND_OK, radios=[R("wlan", 7, 0)]), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", radios=[]), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, radios=[]), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_t in FIXTURES:
        d, reason, t, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, d, reason, t))
        rows.append({"name": name, "decision": d, "reason": reason, "threads": t, "pack": pk.hex(), "close": prev})
        if (d, t) != (want_d, want_t):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-20s %-6s %-14s t=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["threads"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
