#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/power_supply -> i2sd worker power mode.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where power_supply attributes (type, online,
  status, capacity) choose the i2sd worker power mode. Hardware classes: SBC
  arm64 (JuniorOS T4, battery/UPS HAT) and SFF host (JuniorHome T0, mains or
  UPS). Not T13, not item 2 (no GGUF), not item 4 (no Asahi/MLX), not item 5.
  No other team file or earlier hwsw_* seam is imported, read, or wrapped;
  power_supply was not covered before.

What it emulates
  In-memory fixtures shaped like /sys/class/power_supply/<name>/ files: type
  ("Mains", "Battery", "UPS"), online (0/1), status ("Charging",
  "Discharging", "Full", "Not charging"), capacity (0-100). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Unknown type/status, online not 0/1, or capacity not an int 0-100
     -> hold (unparsed).
  4. Mains online=1 -> admit (full_power): 4 workers.
  5. Battery/UPS discharging with capacity <= 10 -> hold (low_battery):
     flush spool and stop accepting.
  6. Battery/UPS discharging with capacity <= 30 -> soft (eco): 1 worker.
  7. Battery/UPS discharging above 30 -> admit (battery): 2 workers.
  8. Battery/UPS not discharging (Charging/Full/Not charging) -> admit
     (full_power): 4 workers.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no power device opened, no suspend or
  shutdown call, no udev, no systemd, no sockets (binds are strings), no
  network fetch, no subprocess, no model pull, no GGUF download. Rows live in
  memory only. Flags mint, socket, layer_one, fetch, ue5_launch all False.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_power_supply_battery_i2sd_worker_mode_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_power_supply_battery_i2sd_worker_mode_seam.py demo   # one line per fixture row
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
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
TYPES = ("Mains", "Battery", "UPS")
STATUSES = ("Charging", "Discharging", "Full", "Not charging")


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
    """Return (decision, reason, workers, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASSES:
        return "hold", "bad_class", 0, [2, 2, 0]
    t, on, st, cap = fx.get("type"), fx.get("online"), fx.get("status"), fx.get("capacity")
    cap_ok = isinstance(cap, int) and not isinstance(cap, bool) and 0 <= cap <= 100
    if t not in TYPES or on not in (0, 1) or st not in STATUSES or not cap_ok:
        return "hold", "unparsed", 0, [2, 2, 2, 0]
    if t == "Mains" and on == 1:
        return "admit", "full_power", 4, [2, 2, 2, 2]
    if st == "Discharging":
        if cap <= 10:
            return "hold", "low_battery", 0, [2, 2, 2, 0]
        if cap <= 30:
            return "soft", "eco", 1, [2, 2, 2, 1]
        return "admit", "battery", 2, [2, 2, 2, 2]
    return "admit", "full_power", 4, [2, 2, 2, 2]


def P(t, on, st, cap):
    return dict(type=t, online=on, status=st, capacity=cap)


FIXTURES = [
    ("sff_mains", dict(cls="sff", bind=BIND_OK, **P("Mains", 1, "Charging", 100)), "admit", 4),
    ("sbc_charging", dict(cls="sbc", bind=BIND_OK, **P("Battery", 1, "Charging", 40)), "admit", 4),
    ("sbc_full", dict(cls="sbc", bind=BIND_OK, **P("Battery", 0, "Full", 100)), "admit", 4),
    ("sbc_battery_80", dict(cls="sbc", bind=BIND_OK, **P("Battery", 0, "Discharging", 80)), "admit", 2),
    ("sff_ups_31", dict(cls="sff", bind=BIND_OK, **P("UPS", 0, "Discharging", 31)), "admit", 2),
    ("sbc_eco_30", dict(cls="sbc", bind=BIND_OK, **P("Battery", 0, "Discharging", 30)), "soft", 1),
    ("sff_ups_low_10", dict(cls="sff", bind=BIND_OK, **P("UPS", 0, "Discharging", 10)), "hold", 0),
    ("bad_capacity", dict(cls="sbc", bind=BIND_OK, **P("Battery", 0, "Discharging", 140)), "hold", 0),
    ("bad_status", dict(cls="sbc", bind=BIND_OK, **P("Battery", 0, "Exploding", 50)), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, **P("Mains", 1, "Charging", 100)), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", **P("Mains", 1, "Charging", 100)), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, **P("Mains", 1, "Charging", 100)), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_w in FIXTURES:
        d, reason, w, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, d, reason, w))
        rows.append({"name": name, "decision": d, "reason": reason, "workers": w, "pack": pk.hex(), "close": prev})
        if (d, w) != (want_d, want_w):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-16s %-6s %-12s workers=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["workers"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
