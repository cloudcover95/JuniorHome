#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/cmdline kernel parameters -> bitnetd admit plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 2, "GGUF - operator sets JUNIOR_GGUF. Bots
  do not download.", at the seam where the kernel command line decides
  whether bitnetd.service on 127.0.0.1:8765 may mmap an operator-provided
  GGUF. Hardware classes: SBC arm64 (JuniorOS T4) and SFF host (JuniorHome
  T0). Not T13, not item 3, not item 4 (no Asahi/MLX), not item 5. No other
  team file or earlier hwsw_* seam is imported, read, or wrapped; no
  /proc/cmdline seam existed before.

What it emulates
  In-memory fixture strings shaped like /proc/cmdline (space separated
  key or key=value tokens). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8765, else reject (bind_reject).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Empty cmdline or a token with an empty key (e.g. "=x") -> hold (unparsed).
  4. cgroup_disable containing "memory" -> hold (no_memcg): bitnetd's
     MemoryMax cannot be enforced.
  5. mem=<N>M/G below the GGUF need (gguf_mb, default 1200) -> hold (mem_cap).
  6. transparent_hugepage=never or hugepages=0 on SFF is fine; on SBC with
     "nohugepages" -> soft (no_thp): admit but half threads.
  7. isolcpus=<list> -> workers use only the non-isolated count: soft
     (isolcpus) with threads = cpus minus isolated, min 1.
  8. Otherwise admit (ok).
  Threads: SBC 4, SFF 8, capped by wanted; soft rows as stated; hold or
  reject gives 0. Rows pack decision codes as 2-bit trits (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no bootloader or cmdline writes, no
  systemd, no sockets (binds are strings), no device, no network fetch, no
  subprocess, no model pull, no GGUF download. Rows live in memory only.
  Flags mint, socket, layer_one, fetch, ue5_launch all False. JuniorSolana
  and the profile repo are untouched.

How to run
  python3 scripts/hwsw_proc_cmdline_bitnetd_gguf_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_proc_cmdline_bitnetd_gguf_admit_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8765"
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


def parse(cmdline):
    """Return dict of key -> value ('' for bare keys), or None if unparsed."""
    toks = cmdline.split()
    if not toks:
        return None
    out = {}
    for t in toks:
        k, _, v = t.partition("=")
        if not k:
            return None
        out[k] = v
    return out


def mem_mb(v):
    try:
        if v[-1:] in "gG":
            return int(v[:-1]) * 1024
        if v[-1:] in "mM":
            return int(v[:-1])
    except ValueError:
        pass
    return None


def isolated_count(v):
    n = 0
    try:
        for part in v.split(","):
            if "-" in part:
                a, b = part.split("-")
                n += int(b) - int(a) + 1
            else:
                int(part)
                n += 1
    except ValueError:
        return None
    return n


def decide(fx):
    """Return (decision, reason, threads, codes)."""
    def done(d, r, codes, t=None):
        if t is None:
            t = 0
            if d in ("admit", "soft"):
                t = min(CLASS_THREADS[fx["cls"]], fx.get("want", 99))
                if d == "soft":
                    t = max(1, t // 2)
        return d, r, t, [DECISION_CODE[d]] + codes

    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("cls") not in CLASS_THREADS:
        return "hold", "bad_class", 0, [2, 0]
    kv = parse(fx.get("cmdline", ""))
    if kv is None:
        return done("hold", "unparsed", [2, 2, 0])
    if "memory" in kv.get("cgroup_disable", "").split(","):
        return done("hold", "no_memcg", [2, 2, 0])
    if "mem" in kv:
        m = mem_mb(kv["mem"])
        if m is None:
            return done("hold", "unparsed", [2, 2, 0])
        if m < fx.get("gguf_mb", 1200):
            return done("hold", "mem_cap", [2, 2, 0])
    if "isolcpus" in kv:
        n = isolated_count(kv["isolcpus"])
        if n is None:
            return done("hold", "unparsed", [2, 2, 0])
        usable = max(1, fx.get("cpus", CLASS_THREADS[fx["cls"]]) - n)
        t = min(usable, CLASS_THREADS[fx["cls"]], fx.get("want", 99))
        return done("soft", "isolcpus", [2, 2, 1], t)
    if fx["cls"] == "sbc" and "nohugepages" in kv:
        return done("soft", "no_thp", [2, 2, 1])
    return done("admit", "ok", [2, 2, 2])


OK_SBC = "console=ttyAMA0,115200 root=/dev/mmcblk0p2 rootwait quiet"
FIXTURES = [
    ("sbc_plain", dict(cls="sbc", bind=BIND_OK, want=8, cmdline=OK_SBC), "admit", 4),
    ("sff_plain", dict(cls="sff", bind=BIND_OK, want=8, cmdline="BOOT_IMAGE=/vmlinuz root=UUID=x ro quiet splash"), "admit", 8),
    ("sff_thp_never", dict(cls="sff", bind=BIND_OK, want=3, cmdline="ro transparent_hugepage=never"), "admit", 3),
    ("sbc_nohugepages", dict(cls="sbc", bind=BIND_OK, want=8, cmdline=OK_SBC + " nohugepages"), "soft", 2),
    ("sff_isolcpus", dict(cls="sff", bind=BIND_OK, want=8, cpus=8, cmdline="ro isolcpus=2-5"), "soft", 4),
    ("sbc_isolcpus_all", dict(cls="sbc", bind=BIND_OK, want=8, cpus=4, cmdline="isolcpus=0-3"), "soft", 1),
    ("no_memcg", dict(cls="sff", bind=BIND_OK, cmdline="ro cgroup_disable=memory"), "hold", 0),
    ("mem_cap_small", dict(cls="sbc", bind=BIND_OK, cmdline="ro mem=512M"), "hold", 0),
    ("mem_cap_ok", dict(cls="sbc", bind=BIND_OK, want=2, cmdline="ro mem=2G"), "admit", 2),
    ("empty_cmdline", dict(cls="sff", bind=BIND_OK, cmdline=""), "hold", 0),
    ("empty_key", dict(cls="sff", bind=BIND_OK, cmdline="ro =x"), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, cmdline="ro"), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8765", cmdline="ro"), "reject", 0),
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
