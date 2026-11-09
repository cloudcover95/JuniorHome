#!/usr/bin/env python3
"""Hardware-to-software seam emulator: hugepages / THP / swap posture -> i2sd userspace arena plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where kernel memory-page posture (HugePages_*
  lines as in /proc/meminfo, the transparent_hugepage enabled mode string,
  vm.swappiness, swap presence, vm.overcommit_memory) decides how i2sd sizes
  its tile arena and whether it may admit. Hardware classes: SBC arm64
  (<=12 W, JuniorOS T4, 4 KiB or 16 KiB base page) and SFF host (JuniorHome
  T0). Distinct from the meminfo/cgroup GGUF seam (no MemAvailable or
  cgroup math), the cache-topology tile seam (no cache sizes), and the
  hwmon/thermal/power seams. Not T13 (overnight owns it). Not item 2 (no
  GGUF), not item 4 (no Asahi/MLX), not item 5 (ue5_launch False).
  No other team file or earlier hwsw_* seam is imported, read, or wrapped.

What it emulates
  In-memory fixtures shaped like the kernel's text: a meminfo-style
  hugepage block, the bracketed THP mode ("always [madvise] never"), and
  integer sysctl values. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767; else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff"; else hold (bad_class).
  3. THP mode parse: the bracketed word must be always/madvise/never; else
     hold (unparsed). "always" -> soft (thp_always: latency spikes from
     compaction); madvise or never is fine.
  4. vm.overcommit_memory 2 (strict) with arena request above the commit
     headroom -> hold (strict_commit). Unparsed -> hold (unparsed).
  5. Swap: swappiness > 60 on SBC -> soft (swappy_sbc); swap present with
     swappiness 0 is fine.
  6. Hugepages: HugePages_Free * Hugepagesize covers the arena request ->
     plan "huge" (arena rounded up to the huge page); else plan "base"
     (rounded up to the base page). Total 0 is fine, plan base. Free >
     Total -> hold (hp_inconsistent). Unparsed -> hold (unparsed).
  Arena request is in MiB, rounded to the page size chosen. Decision is the
  worst row; threads are not decided here (arena only).
  Each row packs decision and check states as 2-bit trit codes (0/1/2,
  four per byte, I2_S order: first code in the low bits); rows chain with
  close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sysctl writes, no hugetlbfs mount, no
  mmap, no madvise, no systemd calls, no sockets (binds are strings;
  nothing listens or connects), no device opened, no network fetch, no
  subprocess, no model pull, no GGUF download. Rows go to a tempfile
  removed after the run; nothing under ~/.juniorhome. Flags mint, socket,
  layer_one, fetch, ue5_launch all False. JuniorSolana and the profile
  repo are untouched.

How to run
  python3 scripts/hwsw_hugepages_thp_i2sd_arena_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_hugepages_thp_i2sd_arena_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
RANK = {"admit": 0, "soft": 1, "hold": 2, "reject": 3}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes: list) -> bytes:
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


def kv(block: str, key: str):
    m = re.search(r"^%s:\s*(\d+)(?:\s*kB)?\s*$" % re.escape(key), block, re.M)
    return int(m.group(1)) if m else None


def thp_mode(s):
    m = re.search(r"\[(\w+)\]", s or "")
    return m.group(1) if m and m.group(1) in ("always", "madvise", "never") else None


def integer(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def round_up(n: int, unit: int) -> int:
    return -(-n // unit) * unit


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    checks, worst, flag = {}, "admit", None

    def mark(name, state, f=None):
        nonlocal worst, flag
        checks[name] = state
        if RANK[state] > RANK[worst]:
            worst, flag = state, f
        elif state != "admit" and RANK[state] == RANK[worst] and flag is None:
            flag = f

    if fx.get("bind") != BIND_OK:
        mark("bind", "reject", "bind_reject")
    elif fx.get("docker_sock"):
        mark("bind", "reject", "docker_sock")
    else:
        mark("bind", "admit")
    mark("cls", "admit" if cls in CLASSES else "hold", None if cls in CLASSES else "bad_class")

    mode = thp_mode(fx.get("thp"))
    if mode is None:
        mark("thp", "hold", "unparsed")
    elif mode == "always":
        mark("thp", "soft", "thp_always")
    else:
        mark("thp", "admit")

    req_kib = int(fx.get("arena_mib", 0)) * 1024
    oc = integer(fx.get("overcommit"))
    headroom = fx.get("commit_headroom_mib")
    if oc is None or oc not in (0, 1, 2):
        mark("commit", "hold", "unparsed")
    elif oc == 2 and headroom is not None and req_kib > int(headroom) * 1024:
        mark("commit", "hold", "strict_commit")
    else:
        mark("commit", "admit")

    sw = integer(fx.get("swappiness"))
    if sw is None:
        mark("swap", "hold", "unparsed")
    elif cls == "sbc" and sw > 60:
        mark("swap", "soft", "swappy_sbc")
    else:
        mark("swap", "admit")

    blk = fx.get("meminfo", "")
    total, free = kv(blk, "HugePages_Total"), kv(blk, "HugePages_Free")
    hp_kib, base_kib = kv(blk, "Hugepagesize"), int(fx.get("base_page_kib", 4))
    plan, arena_kib = "base", round_up(req_kib, base_kib) if req_kib else 0
    if None in (total, free, hp_kib):
        mark("huge", "hold", "unparsed")
    elif free > total:
        mark("huge", "hold", "hp_inconsistent")
    else:
        mark("huge", "admit")
        if free * hp_kib >= round_up(req_kib, hp_kib) > 0:
            plan, arena_kib = "huge", round_up(req_kib, hp_kib)
    if worst in ("hold", "reject"):
        plan, arena_kib = "none", 0
    trits = [DECISION_CODE[worst]] + [DECISION_CODE[checks[k]] for k in
                                      ("bind", "cls", "thp", "commit", "swap", "huge")]
    return {"name": fx["name"], "cls": cls, "decision": worst, "admit": worst == "admit",
            "hold": flag, "plan": plan, "arena_kib": arena_kib, "checks": checks, "trits": trits}


def hp(total, free, size=2048):
    return ("HugePages_Total:%8d\nHugePages_Free:%9d\nHugePages_Rsvd:        0\n"
            "Hugepagesize:%10d kB\n" % (total, free, size))


FIXTURES = [
    {"name": "sff_huge_plan", "cls": "sff", "bind": BIND_OK, "thp": "always [madvise] never",
     "overcommit": "0", "swappiness": "10", "arena_mib": 512, "meminfo": hp(512, 400),
     "want": {"decision": "admit", "plan": "huge", "arena_kib": 524288, "hold": None}},
    {"name": "sff_huge_short_base", "cls": "sff", "bind": BIND_OK, "thp": "always [madvise] never",
     "overcommit": "0", "swappiness": "10", "arena_mib": 512, "meminfo": hp(512, 100),
     "want": {"decision": "admit", "plan": "base", "arena_kib": 524288}},
    {"name": "sbc_no_huge_16k_base", "cls": "sbc", "bind": BIND_OK, "base_page_kib": 16,
     "thp": "always madvise [never]", "overcommit": "0", "swappiness": "30", "arena_mib": 100,
     "meminfo": hp(0, 0),
     "want": {"decision": "admit", "plan": "base", "arena_kib": 102400}},
    {"name": "sbc_odd_arena_rounds_16k", "cls": "sbc", "bind": BIND_OK, "base_page_kib": 16,
     "thp": "always madvise [never]", "overcommit": "1", "swappiness": "0", "arena_mib": 0,
     "meminfo": hp(0, 0), "want": {"decision": "admit", "plan": "base", "arena_kib": 0}},
    {"name": "sff_thp_always_soft", "cls": "sff", "bind": BIND_OK, "thp": "[always] madvise never",
     "overcommit": "0", "swappiness": "10", "arena_mib": 256, "meminfo": hp(0, 0),
     "want": {"decision": "soft", "hold": "thp_always", "plan": "base"}},
    {"name": "sbc_swappy_soft", "cls": "sbc", "bind": BIND_OK, "thp": "always [madvise] never",
     "overcommit": "0", "swappiness": "100", "arena_mib": 64, "meminfo": hp(0, 0),
     "want": {"decision": "soft", "hold": "swappy_sbc"}},
    {"name": "sff_swappy_ok", "cls": "sff", "bind": BIND_OK, "thp": "always [madvise] never",
     "overcommit": "0", "swappiness": "100", "arena_mib": 64, "meminfo": hp(0, 0),
     "want": {"decision": "admit", "hold": None}},
    {"name": "sff_strict_commit_hold", "cls": "sff", "bind": BIND_OK,
     "thp": "always [madvise] never", "overcommit": "2", "commit_headroom_mib": 200,
     "swappiness": "10", "arena_mib": 512, "meminfo": hp(0, 0),
     "want": {"decision": "hold", "hold": "strict_commit", "plan": "none", "arena_kib": 0}},
    {"name": "sff_strict_commit_fits", "cls": "sff", "bind": BIND_OK,
     "thp": "always [madvise] never", "overcommit": "2", "commit_headroom_mib": 1024,
     "swappiness": "10", "arena_mib": 512, "meminfo": hp(0, 0),
     "want": {"decision": "admit", "plan": "base"}},
    {"name": "sff_hp_inconsistent_hold", "cls": "sff", "bind": BIND_OK,
     "thp": "always [madvise] never", "overcommit": "0", "swappiness": "10", "arena_mib": 64,
     "meminfo": hp(10, 12), "want": {"decision": "hold", "hold": "hp_inconsistent"}},
    {"name": "sff_unparsed_thp_hold", "cls": "sff", "bind": BIND_OK, "thp": "always madvise never",
     "overcommit": "0", "swappiness": "10", "arena_mib": 64, "meminfo": hp(0, 0),
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sbc_unparsed_swappiness_hold", "cls": "sbc", "bind": BIND_OK,
     "thp": "always [madvise] never", "overcommit": "0", "swappiness": "n/a", "arena_mib": 64,
     "meminfo": hp(0, 0), "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sff_missing_hugepagesize_hold", "cls": "sff", "bind": BIND_OK,
     "thp": "always [madvise] never", "overcommit": "0", "swappiness": "10", "arena_mib": 64,
     "meminfo": "HugePages_Total:        4\nHugePages_Free:         4\n",
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sbc_wildcard_bind_reject", "cls": "sbc", "bind": "0.0.0.0:8767",
     "thp": "always [madvise] never", "overcommit": "0", "swappiness": "10", "arena_mib": 64,
     "meminfo": hp(0, 0), "want": {"decision": "reject", "admit": False, "hold": "bind_reject"}},
    {"name": "sff_docker_sock_reject", "cls": "sff", "bind": BIND_OK, "docker_sock": True,
     "thp": "always [madvise] never", "overcommit": "0", "swappiness": "10", "arena_mib": 64,
     "meminfo": hp(0, 0), "want": {"decision": "reject", "hold": "docker_sock"}},
    {"name": "bad_class_hold", "cls": "laptop", "bind": BIND_OK, "thp": "always [madvise] never",
     "overcommit": "0", "swappiness": "10", "arena_mib": 64, "meminfo": hp(0, 0),
     "want": {"decision": "hold", "hold": "bad_class", "plan": "none"}},
]


def run(verbose: bool = False) -> dict:
    prev, rows, fails = GENESIS, [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_hugepages_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for fx in FIXTURES:
                r = evaluate(fx)
                text = json.dumps({k: v for k, v in r.items() if k != "trits"}, sort_keys=True)
                prev = close(prev, pack2(r["trits"]), text)
                r["close"] = prev
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                if not all(r.get(k) == v for k, v in fx["want"].items()):
                    fails.append({"name": fx["name"], "got": {k: r.get(k) for k in fx["want"]},
                                  "want": fx["want"]})
                rows.append(r)
                if verbose:
                    print(f"{r['name']:<30} decision={r['decision']:<7} hold={r['hold']!s:<16} "
                          f"plan={r['plan']:<5} close={prev}")
    finally:
        os.remove(path)
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    return {"seam": "hugepages_thp_swap->i2sd_arena",
            "roadmap": "ROADMAP #3 i2sd userspace 127.0.0.1:8767 no docker.sock",
            "rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
            "pack_order_ok": pack_ok, "fail": fails,
            "pass": not fails and pack_ok and not any(FLAGS.values()), "tip_close": prev}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    if not demo:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")}, indent=2))
    sys.exit(0 if out["pass"] else 1)
