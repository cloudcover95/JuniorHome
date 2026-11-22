#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/devices/system/node -> i2sd arena placement.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3: i2sd userspace on 127.0.0.1:8767 (no
  docker.sock). The seam decides, from NUMA topology alone, how the i2sd
  arena may be placed: "single" (one memory node, nothing to place),
  "pin" (bind the arena and workers to one node that has room), "spread"
  (no one node fits, but the online nodes together do; interleave), or
  "hold" (leave placement unchanged and report). SBC boards (JuniorOS T4)
  are almost always one node; SFF hosts (JuniorHome T0) may have several.
  Not ROADMAP T13 (overnight owns it). Not item 5 (ue5_launch stays False).

What it emulates
  In-memory fixtures shaped like /sys/devices/system/node/online,
  node<N>/meminfo (MemTotal, MemFree in kB), node<N>/cpulist and
  node<N>/distance, plus a proposed arena size in MiB and a bind string.
  Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767; anything else -> "reject".
  2. Arena size unparsed or outside 16..4096 MiB -> "hold".
  3. Online mask unparsed, or any node field unparsed -> "hold".
  4. One online node -> "single".
  5. Distance matrix must be square, diagonal 10, symmetric; else "hold".
  6. A node qualifies if MemFree >= 1.25 x arena (25 percent headroom) and
     its cpulist is non-empty (a memory-only node cannot run workers).
     Qualifying nodes are ranked by most free memory, then lowest id; the
     first is chosen -> "pin" (flag far_node if its mean distance to the
     other online nodes exceeds 20).
  7. No node qualifies but summed MemFree of nodes with a cpulist >= 1.25 x
     arena -> "spread" (flag cross_node_cost). Otherwise "hold".
  Each row packs its decision and three check states as real 2-bit trit
  codes (0/1/2, four per byte, I2_S order: first code in the low bits), and
  rows are chained with close rule sha256(prev || pack || text)[:16] at
  rate 0.05.

What it does not touch
  No real /sys reads, no mbind, set_mempolicy, numactl or sched_setaffinity,
  no systemctl, no unit file writes, no sockets (binds are strings; nothing
  listens or connects), no device opened, no model pull, no GGUF, no network
  fetch, no subprocess. It does not import or re-run any team file or any
  earlier hwsw_* seam. Rows go to a tempfile removed after the run; nothing
  is written under ~/.juniorhome. Flags: mint, socket, layer_one, fetch,
  ue5_launch all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_numa_node_i2sd_arena_placement_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_numa_node_i2sd_arena_placement_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
BIND = "127.0.0.1:8767"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
HEADROOM = 1.25
DECISION_CODE = {"hold": 0, "single": 1, "spread": 1, "pin": 2}
CHECK_FAIL, CHECK_SKIP, CHECK_PASS = 0, 1, 2


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
    return [(data[k // 4] >> (2 * (k % 4))) & 3 for k in range(n)]


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_list(s):
    """Parse a kernel list like '0-3,8' into a sorted id list; None if bad."""
    try:
        ids = set()
        for part in str(s).strip().split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                if b < a:
                    return None
                ids.update(range(a, b + 1))
            else:
                ids.add(int(part))
        return sorted(ids)
    except ValueError:
        return None


def decide(fx):
    """Return (decision, checks[bind, topology, fit], flags, node)."""
    if fx.get("bind") != BIND:
        return "reject", [CHECK_FAIL, CHECK_SKIP, CHECK_SKIP], ["bad_bind"], None
    arena = _int(fx.get("arena_mib"))
    if arena is None or not 16 <= arena <= 4096:
        return "hold", [CHECK_PASS, CHECK_SKIP, CHECK_FAIL], ["arena_unparsed_or_range"], None
    online = parse_list(fx.get("online"))
    if not online:
        return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["online_unparsed"], None
    nodes = fx.get("nodes", {})
    info = {}
    for n in online:
        nd = nodes.get(n)
        if nd is None:
            return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["node_missing"], None
        free = _int(nd.get("memfree_kb"))
        total = _int(nd.get("memtotal_kb"))
        cpus = parse_list(nd.get("cpulist", ""))
        if free is None or total is None or cpus is None or free > total:
            return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["node_unparsed"], None
        info[n] = (free / 1024.0, cpus)
    if len(online) == 1:
        return "single", [CHECK_PASS, CHECK_PASS, CHECK_PASS if info[online[0]][0] >= HEADROOM * arena else CHECK_FAIL], \
            ([] if info[online[0]][0] >= HEADROOM * arena else ["low_free_mem"]), online[0]
    dist = {}
    for n in online:
        row = [_int(x) for x in str(nodes[n].get("distance", "")).split()]
        if len(row) != len(online) or any(x is None for x in row):
            return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["distance_unparsed"], None
        dist[n] = row
    for i, n in enumerate(online):
        if dist[n][i] != 10 or any(dist[m][i] != dist[n][j] for j, m in enumerate(online)):
            return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_SKIP], ["distance_bad"], None
    need = HEADROOM * arena
    ok = sorted((n for n in online if info[n][0] >= need and info[n][1]),
                key=lambda n: (-info[n][0], n))
    if ok:
        n = ok[0]
        i = online.index(n)
        others = [d for j, d in enumerate(dist[n]) if j != i]
        flags = ["far_node"] if sum(others) / len(others) > 20 else []
        return "pin", [CHECK_PASS, CHECK_PASS, CHECK_PASS], flags, n
    if sum(info[n][0] for n in online if info[n][1]) >= need:
        return "spread", [CHECK_PASS, CHECK_PASS, CHECK_SKIP], ["cross_node_cost"], None
    return "hold", [CHECK_PASS, CHECK_PASS, CHECK_FAIL], ["insufficient_free_mem"], None


def _node(free_mib, total_mib=16384, cpus="0-7", dist="10"):
    return {"memfree_kb": str(int(free_mib * 1024)), "memtotal_kb": str(total_mib * 1024),
            "cpulist": cpus, "distance": dist}


FIXTURES = [
    ("sbc_one_node_single", {"cls": "SBC", "bind": BIND, "arena_mib": "256", "online": "0",
                             "nodes": {0: _node(3000, 8192, "0-3")}}, "single", 0),
    ("sbc_one_node_low_free", {"cls": "SBC", "bind": BIND, "arena_mib": "256", "online": "0",
                               "nodes": {0: _node(200, 4096, "0-3")}}, "single", 0),
    ("sff_two_node_pin_most_free", {"cls": "SFF", "bind": BIND, "arena_mib": "512", "online": "0-1",
                                    "nodes": {0: _node(900, dist="10 21"), 1: _node(4000, dist="21 10")}}, "pin", 1),
    ("sff_tie_lowest_id", {"cls": "SFF", "bind": BIND, "arena_mib": "512", "online": "0-1",
                           "nodes": {0: _node(4000, dist="10 21"), 1: _node(4000, dist="21 10")}}, "pin", 0),
    ("sff_memory_only_node_skipped", {"cls": "SFF", "bind": BIND, "arena_mib": "512", "online": "0-1",
                                      "nodes": {0: _node(1000, dist="10 21"), 1: _node(9000, cpus="", dist="21 10")}}, "pin", 0),
    ("sff_far_node_flag", {"cls": "SFF", "bind": BIND, "arena_mib": "256", "online": "0-2",
                           "nodes": {0: _node(500, dist="10 21 32"), 1: _node(500, dist="21 10 21"),
                                     2: _node(2000, dist="32 21 10")}}, "pin", 2),
    ("sff_spread_no_single_fit", {"cls": "SFF", "bind": BIND, "arena_mib": "1024", "online": "0-1",
                                  "nodes": {0: _node(800, dist="10 21"), 1: _node(900, dist="21 10")}}, "spread", None),
    ("sff_insufficient_hold", {"cls": "SFF", "bind": BIND, "arena_mib": "2048", "online": "0-1",
                               "nodes": {0: _node(800, dist="10 21"), 1: _node(900, dist="21 10")}}, "hold", None),
    ("sff_asymmetric_distance_hold", {"cls": "SFF", "bind": BIND, "arena_mib": "256", "online": "0-1",
                                      "nodes": {0: _node(2000, dist="10 21"), 1: _node(2000, dist="30 10")}}, "hold", None),
    ("sff_node_unparsed_hold", {"cls": "SFF", "bind": BIND, "arena_mib": "256", "online": "0-1",
                                "nodes": {0: _node(2000, dist="10 21"),
                                          1: {"memfree_kb": "x", "memtotal_kb": "1", "cpulist": "0", "distance": "21 10"}}}, "hold", None),
    ("online_missing_hold", {"cls": "SFF", "bind": BIND, "arena_mib": "256", "online": "", "nodes": {}}, "hold", None),
    ("arena_out_of_range_hold", {"cls": "SFF", "bind": BIND, "arena_mib": "8", "online": "0", "nodes": {0: _node(2000)}}, "hold", None),
    ("wildcard_bind_reject", {"cls": "SFF", "bind": "0.0.0.0:8767", "arena_mib": "256", "online": "0",
                              "nodes": {0: _node(2000)}}, "reject", None),
    ("wrong_port_reject", {"cls": "SBC", "bind": "127.0.0.1:8765", "arena_mib": "256", "online": "0",
                           "nodes": {0: _node(2000)}}, "reject", None),
]


def run_rows():
    rows, prev = [], "0" * 16
    for name, fx, _want, _node_id in FIXTURES:
        dec, checks, flags, node = decide(fx)
        codes = [DECISION_CODE.get(dec, 0)] + checks
        pack = pack_trits(codes)
        text = json.dumps({"row": name, "cls": fx["cls"], "decision": dec, "flags": flags,
                           "node": node, "rate": RATE}, sort_keys=True)
        prev = close(prev, pack, text)
        rows.append({"row": name, "decision": dec, "codes": codes, "pack": pack.hex(),
                     "flags": flags, "node": node, "close": prev})
    return rows, prev


def self_test():
    failures = []
    rows, tip = run_rows()
    for (name, _fx, want, want_node), r in zip(FIXTURES, rows):
        if r["decision"] != want:
            failures.append(f"{name}: decision {r['decision']} != {want}")
        if want_node is not None and r["node"] != want_node:
            failures.append(f"{name}: node {r['node']} != {want_node}")
        if unpack_trits(bytes.fromhex(r["pack"]), len(r["codes"])) != r["codes"]:
            failures.append(f"{name}: trit pack round-trip failed")
        if r["decision"] == "reject" and "bad_bind" not in r["flags"]:
            failures.append(f"{name}: reject without bad_bind")
    by = {r["row"]: r for r in rows}
    if "far_node" not in by["sff_far_node_flag"]["flags"]:
        failures.append("far_node flag missing")
    if "cross_node_cost" not in by["sff_spread_no_single_fit"]["flags"]:
        failures.append("cross_node_cost flag missing")
    if "low_free_mem" not in by["sbc_one_node_low_free"]["flags"]:
        failures.append("low_free_mem flag missing")
    if parse_list("0-3,8") != [0, 1, 2, 3, 8] or parse_list("3-1") is not None or parse_list("a") is not None:
        failures.append("parse_list wrong")
    rows2, tip2 = run_rows()
    if tip2 != tip:
        failures.append("close chain not deterministic")
    if any(FLAGS.values()):
        failures.append("a guard flag is True")
    fd, path = tempfile.mkstemp(prefix="hwsw_numa_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        with open(path) as fh:
            if sum(1 for _ in fh) != len(rows):
                failures.append("tempfile row count mismatch")
    finally:
        os.remove(path)
    return {"seam": "numa node->i2sd arena placement", "bind": BIND, "rate": RATE, "flags": FLAGS,
            "rows": len(rows), "tip_close": tip, "pass": not failures, "failures": failures}


def main(argv):
    if len(argv) > 1 and argv[1] == "demo":
        rows, _ = run_rows()
        for r in rows:
            print(f"{r['row']:<30} {r['decision']:<6} node={r['node']} pack={r['pack']} "
                  f"flags={','.join(r['flags']) or '-'} close={r['close']}")
        return 0
    out = self_test()
    print(json.dumps(out, sort_keys=True))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
