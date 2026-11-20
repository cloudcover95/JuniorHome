#!/usr/bin/env python3
"""Hardware-to-software seam emulator: sysfs cpu online / isolated / nohz_full cpulists -> i2sd worker CPU affinity set.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the cpulists the kernel exposes
  (/sys/devices/system/cpu/online, isolated, nohz_full) decide which CPUs the
  i2sd workers may be pinned to. Hardware classes: SBC arm64 (JuniorOS T4) and
  SFF host (JuniorHome T0). Not T13, not item 2 (no GGUF), not item 4
  (no Asahi/MLX), not item 5. No other team file or earlier hwsw_* seam is
  imported, read, or wrapped; cpulist online/isolated masks were not covered.

What it emulates
  In-memory fixtures shaped like the text sysfs writes: a cpulist such as
  "0-3,6,8-9" (empty string allowed for isolated/nohz_full). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Any cpulist that does not parse (bad token, reversed range, negative)
     -> hold (unparsed). The online list must be non-empty.
  4. allowed = online minus isolated minus nohz_full minus {cpu0 reserved for
     the supervisor, shell and juniorctl} when more than one CPU is online.
     allowed empty -> hold (no_cpu).
  5. Workers = min(class cap, wanted, len(allowed)); class cap SBC 4, SFF 8.
     If len(allowed) < min(class cap, wanted) the result is soft (clamped),
     otherwise admit. The affinity set is the lowest-numbered CPUs of allowed.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no writes, no sched_setaffinity, no cpuset
  cgroup, no CPU hotplug, no systemd, no sockets (binds are strings), no
  device opened, no network fetch, no subprocess, no model pull, no GGUF
  download. Rows live in memory only. Flags mint, socket, layer_one, fetch,
  ue5_launch all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_cpu_online_isolated_nohz_i2sd_affinity_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_cpu_online_isolated_nohz_i2sd_affinity_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
CLASS_WORKERS = {"sbc": 4, "sff": 8}
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


def parse_cpulist(s):
    """Parse a sysfs cpulist into a set of ints, or None if malformed."""
    s = str(s).strip()
    cpus = set()
    if s == "":
        return cpus
    for tok in s.split(","):
        tok = tok.strip()
        if tok.isdigit():
            cpus.add(int(tok))
            continue
        a, sep, b = tok.partition("-")
        if sep != "-" or not a.isdigit() or not b.isdigit() or int(a) > int(b):
            return None
        cpus.update(range(int(a), int(b) + 1))
    return cpus


def decide(fx):
    """Return (decision, reason, affinity list, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", [], [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", [], [2, 0]
    if fx.get("cls") not in CLASS_WORKERS:
        return "hold", "bad_class", [], [2, 2, 0]
    online = parse_cpulist(fx.get("online", ""))
    iso = parse_cpulist(fx.get("isolated", ""))
    nohz = parse_cpulist(fx.get("nohz_full", ""))
    if online is None or iso is None or nohz is None or not online:
        return "hold", "unparsed", [], [2, 2, 2, 0]
    allowed = online - iso - nohz
    if len(online) > 1:
        allowed.discard(0)
    if not allowed:
        return "hold", "no_cpu", [], [2, 2, 2, 2, 0]
    cap = min(CLASS_WORKERS[fx["cls"]], fx.get("want", 99))
    n = min(cap, len(allowed))
    aff = sorted(allowed)[:n]
    if len(allowed) < cap:
        return "soft", "clamped", aff, [2, 2, 2, 2, 1]
    return "admit", "ok", aff, [2, 2, 2, 2, 2]


FIXTURES = [
    ("sbc_four_online", dict(cls="sbc", bind=BIND_OK, want=8, online="0-3"), "soft", [1, 2, 3]),
    ("sbc_eight_online", dict(cls="sbc", bind=BIND_OK, want=8, online="0-7"), "admit", [1, 2, 3, 4]),
    ("sff_sixteen_online", dict(cls="sff", bind=BIND_OK, want=8, online="0-15"), "admit", [1, 2, 3, 4, 5, 6, 7, 8]),
    ("sff_want_two", dict(cls="sff", bind=BIND_OK, want=2, online="0-15"), "admit", [1, 2]),
    ("sff_isolated_skipped", dict(cls="sff", bind=BIND_OK, want=4, online="0-7", isolated="2-3"), "admit", [1, 4, 5, 6]),
    ("sff_nohz_and_isolated", dict(cls="sff", bind=BIND_OK, want=8, online="0-7", isolated="1-2", nohz_full="3-4"), "soft", [5, 6, 7]),
    ("sbc_hotplug_gap", dict(cls="sbc", bind=BIND_OK, want=4, online="0,2,4-6"), "admit", [2, 4, 5, 6]),
    ("sbc_single_cpu_keeps_cpu0", dict(cls="sbc", bind=BIND_OK, want=4, online="0"), "soft", [0]),
    ("sff_all_isolated", dict(cls="sff", bind=BIND_OK, want=4, online="0-3", isolated="1-3"), "hold", []),
    ("unparsed_reversed_range", dict(cls="sff", bind=BIND_OK, online="7-3"), "hold", []),
    ("unparsed_empty_online", dict(cls="sbc", bind=BIND_OK, online=""), "hold", []),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, online="0-3"), "hold", []),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", online="0-3"), "reject", []),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, online="0-3"), "reject", []),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_aff in FIXTURES:
        d, reason, aff, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%s" % (name, d, reason, ",".join(map(str, aff))))
        rows.append({"name": name, "decision": d, "reason": reason, "affinity": aff, "pack": pk.hex(), "close": prev})
        if (d, aff) != (want_d, want_aff):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    parse_ok = parse_cpulist("0-3,6,8-9") == {0, 1, 2, 3, 6, 8, 9} and parse_cpulist("3-1") is None
    return rows, fails, pack_ok and parse_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-26s %-6s %-12s aff=%s pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["affinity"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
