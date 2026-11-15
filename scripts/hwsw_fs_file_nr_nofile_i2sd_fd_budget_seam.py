#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/sys/fs file-nr + RLIMIT_NOFILE -> i2sd fd budget admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where kernel file-handle pressure (fs.file-nr,
  fs.file-max) and the service soft nofile limit decide how many worker
  threads (one fd each plus listener) i2sd may open. Hardware classes: SBC
  arm64 (JuniorOS T4) and SFF host (JuniorHome T0). Not T13, not item 2, not
  item 4, not item 5. No other team file or earlier hwsw_* seam is imported,
  read, or wrapped; fs.file-nr was not covered before.

What it emulates
  In-memory fixtures shaped like /proc/sys/fs/file-nr ("alloc unused max"
  as one string) and a nofile soft limit. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. file-nr not three non-negative ints, or max == 0, or nofile < 1 -> hold
     (unparsed).
  4. nofile soft < 64 -> hold (nofile_tiny): no room for listener + workers.
  5. System fd use (alloc-unused)/max >= 0.90 -> hold (fd_exhausted).
  6. Use >= 0.75 -> soft (fd_pressure), threads halved (floor, min 1).
  7. nofile soft < 256 -> soft (nofile_low).
  8. Otherwise admit.
  Threads: SBC 4, SFF 8, capped by wanted; hold/reject gives 0. Rows pack
  decision and check states as 2-bit trit codes (four per byte, I2_S order,
  first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sysctl writes, no setrlimit, no systemd,
  no sockets (binds are strings), no subprocess, no network fetch, no model
  pull, no GGUF download. Rows live in memory only. Flags mint, socket,
  layer_one, fetch, ue5_launch all False. JuniorSolana and the profile repo
  are untouched.

How to run
  python3 scripts/hwsw_fs_file_nr_nofile_i2sd_fd_budget_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_fs_file_nr_nofile_i2sd_fd_budget_seam.py demo   # one line per fixture row
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


def parse_file_nr(s):
    parts = str(s).split()
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    a, u, m = (int(p) for p in parts)
    return (a, u, m) if m > 0 and u <= a else None


def decide(fx):
    """Return (decision, reason, threads, codes)."""
    def done(d, r, codes):
        t = 0
        if d in ("admit", "soft"):
            t = min(CLASS_THREADS[fx["cls"]], fx.get("want", 99))
            if d == "soft" and r == "fd_pressure":
                t = max(1, t // 2)
        return d, r, t, [DECISION_CODE[d]] + codes

    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASS_THREADS:
        return "hold", "bad_class", 0, [2, 2, 0]
    fn = parse_file_nr(fx.get("file_nr"))
    nf = fx.get("nofile")
    if fn is None or not isinstance(nf, int) or nf < 1:
        return done("hold", "unparsed", [2, 2, 0])
    if nf < 64:
        return done("hold", "nofile_tiny", [2, 2, 2, 0])
    use = (fn[0] - fn[1]) / fn[2]
    if use >= 0.90:
        return done("hold", "fd_exhausted", [2, 2, 2, 2, 0])
    if use >= 0.75:
        return done("soft", "fd_pressure", [2, 2, 2, 2, 1])
    if nf < 256:
        return done("soft", "nofile_low", [2, 2, 2, 1])
    return done("admit", "ok", [2, 2, 2, 2, 2])


FIXTURES = [
    ("sbc_roomy", dict(cls="sbc", bind=BIND_OK, want=8, file_nr="1900 0 800000", nofile=1024), "admit", 4),
    ("sff_roomy", dict(cls="sff", bind=BIND_OK, want=8, file_nr="5200 100 9000000", nofile=65536), "admit", 8),
    ("sff_pressure", dict(cls="sff", bind=BIND_OK, want=8, file_nr="8000 0 10000", nofile=4096), "soft", 4),
    ("sbc_pressure_one", dict(cls="sbc", bind=BIND_OK, want=1, file_nr="800 0 1000", nofile=1024), "soft", 1),
    ("sbc_nofile_low", dict(cls="sbc", bind=BIND_OK, want=8, file_nr="1000 0 800000", nofile=128), "soft", 4),
    ("sff_exhausted", dict(cls="sff", bind=BIND_OK, want=8, file_nr="9500 0 10000", nofile=4096), "hold", 0),
    ("sbc_nofile_tiny", dict(cls="sbc", bind=BIND_OK, want=8, file_nr="1000 0 800000", nofile=32), "hold", 0),
    ("unparsed_file_nr", dict(cls="sff", bind=BIND_OK, file_nr="12 x 99", nofile=1024), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, file_nr="1 0 100", nofile=1024), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", file_nr="1 0 100", nofile=1024), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, file_nr="1 0 100", nofile=1024), "reject", 0),
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
            print("%-18s %-6s %-13s t=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["threads"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
