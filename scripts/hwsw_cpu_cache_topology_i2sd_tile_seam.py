#!/usr/bin/env python3
"""Hardware-to-software seam emulator: CPU cache + core topology -> i2sd I2_S tile rows and threads.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767. No
  docker.sock." The seam decides, from the CPU cache and core topology the
  kernel exposes, how many I2_S rows the i2sd userspace daemon should put in
  one scoring tile (so a tile's packed weights stay inside a core's share of
  L2) and how many worker threads to run (one per physical core, never one
  per SMT sibling). SBC boards (JuniorOS T4, arm64, often big.LITTLE with
  small or missing per-core L2) and SFF hosts (JuniorHome T0, x86_64 with
  SMT, or Apple-silicon Asahi with a large cluster L2) land on very
  different tiles. Not ROADMAP T13 (overnight owns it). Not item 4 (the
  Asahi fixture here is only a topology shape; no MLX probe). Not item 5
  (ue5_launch stays False).

What it emulates
  In-memory fixtures shaped like /sys/devices/system/cpu/online,
  /sys/devices/system/cpu/cpuN/topology/thread_siblings_list, and
  /sys/devices/system/cpu/cpuN/cache/indexM/{level,type,size,
  coherency_line_size,shared_cpu_list}, plus the model's hidden width
  (columns). One I2_S row costs columns * 2 / 8 bytes. Rules, in order:
  1. Bind: the i2sd bind string must be exactly 127.0.0.1:8767; anything
     else (0.0.0.0, another port) -> "reject".
  2. online, sibling lists and every cache size/line must parse (sizes like
     "512K", "32M"); otherwise -> "hold" (topology_unparsed). No online CPU
     -> "hold" (no_online_cpu).
  3. Any Data/Unified cache line other than 64 or 128 bytes -> "hold"
     (line_unsupported).
  4. Per-core L2 share = L2 size / distinct physical cores in its
     shared_cpu_list (SMT siblings are one core). If online clusters have
     different shares (big.LITTLE), the smallest is used -> "soft"
     (hetero_l2). No L2 at all -> tile 512 and "soft" (no_l2).
  5. Tile rows = largest power of two with rows * row_bytes <= 3/4 of the
     per-core L2 share, clamped to 512..8192. Below 512 is clamped up and
     marked "soft" (tile_floor).
  6. Threads = distinct physical cores among online CPUs, capped at 4 on
     SBC and 8 on SFF. SMT present is noted (smt_one_per_core) and stays
     "ready". Offline CPUs are ignored.
  Each row also checks that a tiled ternary dot over a small fixed matrix
  equals the untiled scalar reference. Each row packs its decision and three
  check states (bind, cache, threads) as 2-bit trit codes (hold/reject 0,
  soft 1, ready 2; four per byte, first code in the low bits), and rows are
  chained with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no sched_setaffinity, no cpuid, no perf
  counters, no device opened, no sockets (binds are strings; nothing
  listens or connects), no model pull, no GGUF, no network fetch, no
  subprocess. It does not import or re-run any team file, bench finding, or
  earlier hwsw_* seam (CPU ISA, thermal, power, memory, and so on); it has
  its own small trit packer. Rows go to a tempfile removed after the run;
  nothing is written under ~/.juniorhome. Flags: mint, socket, layer_one,
  fetch, ue5_launch all False. JuniorSolana and the profile repo are
  untouched.

How to run
  python3 scripts/hwsw_cpu_cache_topology_i2sd_tile_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_cpu_cache_topology_i2sd_tile_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

BIND = "127.0.0.1:8767"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
GENESIS = "0" * 16
CODE = {"reject": 0, "hold": 0, "soft": 1, "ready": 2}
THREAD_CAP = {"sbc": 4, "sff": 8}
TILE_MIN, TILE_MAX = 512, 8192


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack_trits(codes):
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            if c not in (0, 1, 2):
                raise ValueError("trit code must be 0, 1 or 2")
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_trits(data: bytes, n: int):
    codes = []
    for b in data:
        for j in range(4):
            codes.append((b >> (2 * j)) & 3)
    return codes[:n]


def parse_cpulist(s):
    """'0-3,6' -> {0,1,2,3,6}. Raises ValueError on junk."""
    s = str(s).strip()
    if not s:
        return set()
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                raise ValueError("bad range")
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def parse_size(s):
    s = str(s).strip().upper()
    mult = 1
    if s.endswith("K"):
        mult, s = 1024, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 * 1024, s[:-1]
    v = int(s) * mult
    if v <= 0:
        raise ValueError("size")
    return v


def ternary_dot_check(tile_rows):
    """Tiled ternary dot over a fixed 37x24 matrix equals the untiled scalar ref."""
    rows, cols = 37, 24
    w = [[((r * 7 + c * 3) % 3) - 1 for c in range(cols)] for r in range(rows)]
    x = [((c * 5) % 11) - 5 for c in range(cols)]
    ref = [sum(w[r][c] * x[c] for c in range(cols)) for r in range(rows)]
    step = max(1, tile_rows // 512)  # scale the real tile down to the toy matrix
    got = []
    for t0 in range(0, rows, step):
        for r in range(t0, min(rows, t0 + step)):
            acc = 0
            for c in range(cols):
                if w[r][c] == 1:
                    acc += x[c]
                elif w[r][c] == -1:
                    acc -= x[c]
            got.append(acc)
    return got == ref


def decide(fx):
    """Return (decision, flags, tile_rows, threads, codes)."""
    flags = []
    if fx.get("bind") != BIND:
        return "reject", ["bind_not_loopback_8767"], 0, 0, [CODE["reject"], 0, 0, 0]
    try:
        online = parse_cpulist(fx["online"])
        siblings = {cpu: frozenset(parse_cpulist(fx["siblings"][cpu])) for cpu in online}
        caches = []
        for c in fx.get("caches", []):
            caches.append({"level": int(c["level"]), "type": str(c["type"]), "size": parse_size(c["size"]),
                           "line": int(c["line"]), "shared": parse_cpulist(c["shared"])})
        cols = int(fx["cols"])
        if cols <= 0 or cols % 4:
            raise ValueError("cols")
    except (KeyError, ValueError, TypeError):
        return "hold", ["topology_unparsed"], 0, 0, [CODE["hold"], CODE["ready"], 0, 0]
    if not online:
        return "hold", ["no_online_cpu"], 0, 0, [CODE["hold"], CODE["ready"], 0, 0]
    for c in caches:
        if c["type"] in ("Data", "Unified") and c["line"] not in (64, 128):
            return "hold", ["line_unsupported"], 0, 0, [CODE["hold"], CODE["ready"], CODE["hold"], 0]

    def core_of(cpu):
        return siblings.get(cpu, frozenset([cpu]))

    cores = {core_of(cpu) for cpu in online}
    smt = any(len(s & online) > 1 for s in cores)
    cache_state = "ready"
    shares = []
    for c in caches:
        if c["level"] == 2 and c["type"] in ("Data", "Unified") and (c["shared"] & online):
            ncores = len({core_of(cpu) for cpu in c["shared"] if cpu in online}) or 1
            shares.append(c["size"] // ncores)
    row_bytes = cols * 2 // 8
    if not shares:
        tile = TILE_MIN
        flags.append("no_l2")
        cache_state = "soft"
    else:
        if len(set(shares)) > 1:
            flags.append("hetero_l2")
            cache_state = "soft"
        budget = min(shares) * 3 // 4
        tile = 1
        while tile * 2 * row_bytes <= budget:
            tile *= 2
        if tile * row_bytes > budget:
            tile = 0
        if tile < TILE_MIN:
            tile = TILE_MIN
            flags.append("tile_floor")
            cache_state = "soft"
        tile = min(tile, TILE_MAX)
    threads = min(len(cores), THREAD_CAP[fx["cls"]])
    if smt:
        flags.append("smt_one_per_core")
    if not ternary_dot_check(tile):
        return "hold", flags + ["tiled_dot_mismatch"], tile, threads, [CODE["hold"], CODE["ready"], CODE["hold"], 0]
    dec = "soft" if cache_state == "soft" else "ready"
    return dec, flags, tile, threads, [CODE[dec], CODE["ready"], CODE[cache_state], CODE["ready"]]


def _cache(level, typ, size, shared, line=64):
    return {"level": level, "type": typ, "size": size, "line": line, "shared": shared}


def _sib(n, pairs=None):
    if pairs:
        d = {}
        for a, b in pairs:
            d[a] = d[b] = "%d,%d" % (a, b)
        return d
    return {i: str(i) for i in range(n)}


FIXTURES = [
    # Pi 5 style SBC: 4x A76, 512K private L2 each, 2B model width 2560 -> 640 B/row.
    ("sbc_a76_quad", {"cls": "sbc", "bind": BIND, "online": "0-3", "siblings": _sib(4), "cols": 2560,
                      "caches": [_cache(1, "Data", "64K", str(i)) for i in range(4)]
                      + [_cache(2, "Unified", "512K", str(i)) for i in range(4)]
                      + [_cache(3, "Unified", "2048K", "0-3")]},
     "ready", [], 512, 4),
    # RK3588 style SBC: 4x A55 (128K L2) + 4x A76 (512K L2) -> smallest share wins, floor.
    ("sbc_big_little", {"cls": "sbc", "bind": BIND, "online": "0-7", "siblings": _sib(8), "cols": 2560,
                        "caches": [_cache(2, "Unified", "128K", str(i)) for i in range(4)]
                        + [_cache(2, "Unified", "512K", str(i)) for i in range(4, 8)]},
     "soft", ["hetero_l2", "tile_floor"], 512, 4),
    # A53 board with no L2 entry exported.
    ("sbc_no_l2", {"cls": "sbc", "bind": BIND, "online": "0-3", "siblings": _sib(4), "cols": 2560,
                   "caches": [_cache(1, "Data", "32K", str(i)) for i in range(4)]},
     "soft", ["no_l2"], 512, 4),
    # x86 SFF: 8 cores / 16 threads (siblings i, i+8), 1280K L2 per core; cpu 15 offline.
    ("sff_x86_smt", {"cls": "sff", "bind": BIND, "online": "0-14", "siblings": _sib(16, [(i, i + 8) for i in range(8)]),
                     "cols": 2560,
                     "caches": [_cache(2, "Unified", "1280K", "%d,%d" % (i, i + 8)) for i in range(8)]},
     "ready", ["smt_one_per_core"], 1024, 8),
    # Asahi-shaped SFF topology: E-cluster 4M and P-cluster 16M L2 (1M vs 4M per core) -> smallest share, 1024 rows.
    ("sff_asahi_cluster", {"cls": "sff", "bind": BIND, "online": "0-7", "siblings": _sib(8), "cols": 2560,
                           "caches": [_cache(2, "Unified", "4M", "0-3", 128), _cache(2, "Unified", "16M", "4-7", 128)]},
     "soft", ["hetero_l2"], 1024, 8),
    # Unparsed L2 size string.
    ("sff_unparsed_size", {"cls": "sff", "bind": BIND, "online": "0-3", "siblings": _sib(4), "cols": 2560,
                           "caches": [_cache(2, "Unified", "lots", "0-3")]},
     "hold", ["topology_unparsed"], 0, 0),
    # 32-byte line.
    ("sbc_line_32", {"cls": "sbc", "bind": BIND, "online": "0-1", "siblings": _sib(2), "cols": 2560,
                     "caches": [_cache(2, "Unified", "256K", "0-1", 32)]},
     "hold", ["line_unsupported"], 0, 0),
    # Wildcard bind.
    ("sff_bind_wildcard", {"cls": "sff", "bind": "0.0.0.0:8767", "online": "0-3", "siblings": _sib(4), "cols": 2560,
                           "caches": [_cache(2, "Unified", "1M", str(i)) for i in range(4)]},
     "reject", ["bind_not_loopback_8767"], 0, 0),
]


def _text(name, cls, dec, flags, tile, threads):
    return json.dumps({"row": name, "cls": cls, "decision": dec, "flags": flags, "tile_rows": tile,
                       "threads": threads, "rate": RATE}, sort_keys=True)


def run(verbose=False):
    failures = []
    rows = []
    prev = GENESIS
    fd, path = tempfile.mkstemp(prefix="hwsw_cache_topo_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for name, fx, want_dec, want_flags, want_tile, want_threads in FIXTURES:
                dec, flags, tile, threads, codes = decide(fx)
                pack = pack_trits(codes)
                if unpack_trits(pack, len(codes)) != codes:
                    failures.append(name + ":pack_roundtrip")
                cur = close(prev, pack, _text(name, fx["cls"], dec, flags, tile, threads))
                row = {"row": name, "decision": dec, "flags": flags, "tile_rows": tile, "threads": threads,
                       "pack": pack.hex(), "prev": prev, "close": cur}
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
                if (dec, flags, tile, threads) != (want_dec, want_flags, want_tile, want_threads):
                    failures.append("%s: got %s %s %s %s want %s %s %s %s" % (
                        name, dec, flags, tile, threads, want_dec, want_flags, want_tile, want_threads))
                if verbose:
                    print(json.dumps(row, sort_keys=True))
                prev = cur
        with open(path) as fh:
            if sum(1 for _ in fh) != len(FIXTURES):
                failures.append("tempfile_row_count")
    finally:
        os.remove(path)
    p = GENESIS
    for (name, fx, _, _, _, _), r in zip(FIXTURES, rows):
        dec, flags, tile, threads, codes = decide(fx)
        p2 = close(p, pack_trits(codes), _text(name, fx["cls"], dec, flags, tile, threads))
        if p2 != r["close"]:
            failures.append(name + ":chain")
        p = p2
    if any(FLAGS.values()):
        failures.append("flags_not_false")
    return {"seam": "cpu_cache_topology->i2sd_tile_threads", "roadmap": "ROADMAP #3 i2sd userspace", "bind": BIND,
            "rate": RATE, "flags": FLAGS, "rows": len(rows), "tip_close": prev,
            "pass": not failures, "failures": failures}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    result = run(verbose=demo)
    print(json.dumps(result, sort_keys=True))
    sys.exit(0 if result["pass"] else 1)
