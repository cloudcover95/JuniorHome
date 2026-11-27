#!/usr/bin/env python3
"""Hardware-to-software seam emulator: RLIMIT_MEMLOCK / CAP_IPC_LOCK -> i2sd pinned-arena plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the kernel's locked-memory limit (as it
  would read in /proc/<pid>/limits "Max locked memory") decides whether i2sd
  can mlock its receipt arena. Hardware classes: SBC arm64 (JuniorOS T4, small
  RAM, 64 KiB default limit) and SFF host (JuniorHome T0). Not T13, not item 2
  (no GGUF), not item 4 (no Asahi/MLX), not item 5. No other team file or
  earlier hwsw_* seam is imported, read, or wrapped; RLIMIT_MEMLOCK was not
  covered before.

What it emulates
  In-memory fixtures shaped like limits and capability attributes: soft and
  hard memlock limits (int bytes or the string "unlimited"), want_kib (arena
  size i2sd asks to pin), ram_kib, and cap_ipc_lock (0/1). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Limits not int >= 0 or "unlimited", soft above hard, want_kib not a
     positive int, ram_kib not a positive int, or cap_ipc_lock not 0/1 ->
     hold (unparsed).
  4. want_kib above half of ram_kib -> soft (ram_cap): pin half of RAM at
     most, even when the limit allows more.
  5. cap_ipc_lock 1 or soft "unlimited" -> admit (pin_full): pin want_kib.
  6. soft limit covers want_kib -> admit (pin_full).
  7. soft limit is 0 -> soft (no_pin): run unpinned, pin 0 KiB.
  8. otherwise -> soft (pin_partial): pin soft limit in KiB, rest unpinned.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc, /sys or limits reads, no setrlimit, no mlock or mmap call, no
  capability change, no systemd, no sockets (binds are strings), no network
  fetch, no subprocess, no model pull, no GGUF download. Rows live in memory
  only. Flags mint, socket, layer_one, fetch, ue5_launch all False.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_rlimit_memlock_cap_ipc_lock_i2sd_pinned_arena_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_rlimit_memlock_cap_ipc_lock_i2sd_pinned_arena_seam.py demo   # one line per fixture row
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
UNL = "unlimited"
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


def lim_ok(v):
    return v == UNL or (type(v) is int and v >= 0)


def pos(v):
    return type(v) is int and v > 0


def decide(fx):
    """Return (decision, reason, pin_kib, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASSES:
        return "hold", "bad_class", 0, [2, 2, 0]
    soft, hard = fx.get("soft"), fx.get("hard")
    want, ram, cap = fx.get("want_kib"), fx.get("ram_kib"), fx.get("cap_ipc_lock")
    if not lim_ok(soft) or not lim_ok(hard) or not pos(want) or not pos(ram) or not is01(cap):
        return "hold", "unparsed", 0, [2, 2, 2, 0]
    if hard != UNL and (soft == UNL or soft > hard):
        return "hold", "unparsed", 0, [2, 2, 2, 0]
    half = ram // 2
    if want > half:
        return "soft", "ram_cap", half, [2, 2, 2, 1]
    if cap == 1 or soft == UNL:
        return "admit", "pin_full", want, [2, 2, 2, 2]
    soft_kib = soft // 1024
    if soft_kib >= want:
        return "admit", "pin_full", want, [2, 2, 2, 2]
    if soft_kib == 0:
        return "soft", "no_pin", 0, [2, 2, 2, 1]
    return "soft", "pin_partial", soft_kib, [2, 2, 2, 1]


def F(soft, hard, want, ram, cap=0):
    return dict(soft=soft, hard=hard, want_kib=want, ram_kib=ram, cap_ipc_lock=cap)


GIB = 1024 * 1024
FIXTURES = [
    ("sbc_default_64k", dict(cls="sbc", bind=BIND_OK, **F(65536, 65536, 4096, 2 * GIB)), "soft", 64),
    ("sff_unlimited", dict(cls="sff", bind=BIND_OK, **F(UNL, UNL, 8192, 16 * GIB)), "admit", 8192),
    ("sff_8m_limit_ok", dict(cls="sff", bind=BIND_OK, **F(8 * 1048576, UNL, 8192, 16 * GIB)), "admit", 8192),
    ("sbc_zero_limit", dict(cls="sbc", bind=BIND_OK, **F(0, 0, 4096, 2 * GIB)), "soft", 0),
    ("sbc_cap_ipc_lock", dict(cls="sbc", bind=BIND_OK, **F(65536, 65536, 4096, 2 * GIB, 1)), "admit", 4096),
    ("sbc_tiny_ram", dict(cls="sbc", bind=BIND_OK, **F(UNL, UNL, 600000, 1000000)), "soft", 500000),
    ("sff_partial", dict(cls="sff", bind=BIND_OK, **F(2 * 1048576, UNL, 8192, 16 * GIB)), "soft", 2048),
    ("soft_gt_hard", dict(cls="sff", bind=BIND_OK, **F(2048, 1024, 4, GIB)), "hold", 0),
    ("soft_unl_hard_int", dict(cls="sff", bind=BIND_OK, **F(UNL, 4096, 4, GIB)), "hold", 0),
    ("neg_limit", dict(cls="sff", bind=BIND_OK, **F(-1, UNL, 4, GIB)), "hold", 0),
    ("zero_want", dict(cls="sff", bind=BIND_OK, **F(UNL, UNL, 0, GIB)), "hold", 0),
    ("bad_cap", dict(cls="sff", bind=BIND_OK, **F(UNL, UNL, 4, GIB, 3)), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, **F(UNL, UNL, 4, GIB)), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", **F(UNL, UNL, 4, GIB)), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, **F(UNL, UNL, 4, GIB)), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_p in FIXTURES:
        d, reason, p, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, d, reason, p))
        rows.append({"name": name, "decision": d, "reason": reason, "pin_kib": p, "pack": pk.hex(), "close": prev})
        if (d, p) != (want_d, want_p):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-18s %-6s %-12s pin=%dKiB pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["pin_kib"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
