#!/usr/bin/env python3
"""Hardware-to-software seam emulator: cgroup pids.max / kernel threads-max / pid_max -> i2sd worker cap.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the task-count limits the kernel exposes
  (cgroup pids.max and pids.current, /proc/sys/kernel/threads-max, pid_max)
  bound how many worker threads i2sd may spawn. Hardware classes: SBC arm64
  (JuniorOS T4) and SFF host (JuniorHome T0). Not T13, not item 2 (no GGUF),
  not item 4 (no Asahi/MLX), not item 5. No other team file or earlier
  hwsw_* seam is imported, read, or wrapped; pids/threads-max was not covered.

What it emulates
  In-memory fixtures shaped like the text the kernel writes: pids.max is an
  integer or the word "max", pids.current an integer, threads-max and pid_max
  integers. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Any value that is not a non-negative integer (pids.max may be "max") ->
     hold (unparsed).
  4. headroom = min(pids.max or inf, threads-max, pid_max) - pids.current,
     minus a reserve of 8 tasks for the supervisor, shell and juniorctl.
     headroom < 1 -> hold (no_headroom).
  5. Threads = min(class cap, wanted, headroom); class cap SBC 4, SFF 8.
     If headroom < class cap (or wanted) the result is soft (clamped),
     otherwise admit. Soft never goes below 1 thread.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc, /sys or /sys/fs/cgroup reads, no writes, no setrlimit, no
  fork or thread creation, no systemd, no sockets (binds are strings), no
  device opened, no network fetch, no subprocess, no model pull, no GGUF
  download. Rows live in memory only. Flags mint, socket, layer_one, fetch,
  ue5_launch all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_cgroup_pids_max_threads_max_i2sd_worker_cap_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_cgroup_pids_max_threads_max_i2sd_worker_cap_seam.py demo   # one line per fixture row
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
RESERVE = 8
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


def parse_int(s, allow_max=False):
    s = str(s).strip()
    if allow_max and s == "max":
        return float("inf")
    return int(s) if s.isdigit() else None


def decide(fx):
    """Return (decision, reason, threads, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASS_THREADS:
        return "hold", "bad_class", 0, [2, 2, 0]
    pmax = parse_int(fx.get("pids_max", ""), True)
    cur = parse_int(fx.get("pids_current", ""))
    tmax = parse_int(fx.get("threads_max", ""))
    pidmax = parse_int(fx.get("pid_max", ""))
    if None in (pmax, cur, tmax, pidmax):
        return "hold", "unparsed", 0, [2, 2, 2, 0]
    headroom = min(pmax, tmax, pidmax) - cur - RESERVE
    if headroom < 1:
        return "hold", "no_headroom", 0, [2, 2, 2, 2, 0]
    cap = min(CLASS_THREADS[fx["cls"]], fx.get("want", 99))
    t = int(min(cap, headroom))
    if headroom < cap:
        return "soft", "clamped", max(1, t), [2, 2, 2, 2, 1]
    return "admit", "ok", t, [2, 2, 2, 2, 2]


FIXTURES = [
    ("sbc_roomy", dict(cls="sbc", bind=BIND_OK, want=8, pids_max="max", pids_current="40", threads_max="30000", pid_max="32768"), "admit", 4),
    ("sff_roomy", dict(cls="sff", bind=BIND_OK, want=8, pids_max="4915", pids_current="120", threads_max="250000", pid_max="4194304"), "admit", 8),
    ("sff_want_three", dict(cls="sff", bind=BIND_OK, want=3, pids_max="max", pids_current="10", threads_max="250000", pid_max="4194304"), "admit", 3),
    ("sbc_cgroup_tight", dict(cls="sbc", bind=BIND_OK, want=8, pids_max="30", pids_current="22", threads_max="30000", pid_max="32768"), "hold", 0),
    ("sbc_clamped_two", dict(cls="sbc", bind=BIND_OK, want=8, pids_max="40", pids_current="30", threads_max="30000", pid_max="32768"), "soft", 2),
    ("sff_threads_max_low", dict(cls="sff", bind=BIND_OK, want=8, pids_max="max", pids_current="100", threads_max="112", pid_max="4194304"), "soft", 4),
    ("sff_full", dict(cls="sff", bind=BIND_OK, want=8, pids_max="200", pids_current="200", threads_max="250000", pid_max="4194304"), "hold", 0),
    ("unparsed_max_on_threads", dict(cls="sff", bind=BIND_OK, pids_max="max", pids_current="1", threads_max="max", pid_max="4194304"), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, pids_max="max", pids_current="1", threads_max="1000", pid_max="1000"), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", pids_max="max", pids_current="1", threads_max="1000", pid_max="1000"), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, pids_max="max", pids_current="1", threads_max="1000", pid_max="1000"), "reject", 0),
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
            print("%-24s %-6s %-12s t=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["threads"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
