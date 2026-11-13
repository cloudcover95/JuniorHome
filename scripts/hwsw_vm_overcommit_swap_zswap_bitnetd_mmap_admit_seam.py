#!/usr/bin/env python3
"""Hardware-to-software seam emulator: vm sysctl overcommit + swappiness + zswap -> bitnetd GGUF mmap admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 2, "GGUF - operator sets JUNIOR_GGUF. Bots
  do not download.", at the seam where kernel virtual-memory policy
  (/proc/sys/vm/overcommit_memory, overcommit_ratio, swappiness, and the
  zswap enabled flag) decides whether bitnetd.service on 127.0.0.1:8765 may
  mmap the operator's GGUF. Hardware classes: SBC arm64 (JuniorOS T4) and SFF
  host (JuniorHome T0). Distinct from the meminfo, cgroup memory.max /
  memory.high, and io.max seams: this file reads no meminfo and no cgroup
  files. Not T13 (overnight owns it). Not item 3/4/5. No other team file or
  earlier hwsw_* seam is imported, read, or wrapped.

What it emulates
  In-memory fixtures shaped like the sysctl and sysfs text files (integers as
  the kernel writes them), plus class, bind string, GGUF size in MiB, and
  RAM and swap size in MiB. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8765; else reject (bind_reject).
  2. Class must be "sbc" or "sff"; else hold (bad_class).
  3. overcommit_memory: 2 (strict) -> commit limit = swap + RAM*ratio/100;
     GGUF larger than the limit -> hold (commit_limit); unparsed or not in
     0..2 -> hold (unparsed). 1 (always) -> soft (overcommit_always: OOM
     killer risk). 0 (heuristic) -> admit if GGUF <= RAM + swap, else hold
     (heuristic_refuse).
  4. swappiness: unparsed or outside 0..200 -> hold (unparsed). Above 60
     with GGUF > 50% RAM -> soft (swappy_weights, paging out model pages).
  5. zswap: "Y" or "1" with SBC -> soft (zswap_cpu_cost); "N"/"0"/absent
     is fine; any other text -> hold (unparsed).
  6. No GGUF (size 0, operator has not set JUNIOR_GGUF) -> hold (no_gguf).
  Worst state wins. Rows pack decision and check states as 2-bit trit codes
  (0/1/2, four per byte, I2_S order, first code in the low bits) and chain
  with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sysctl writes, no mmap, no swapon, no
  subprocess, no sockets (the bind is a string; nothing listens), no device
  opened, no network fetch, no model pull, no GGUF download (sizes are plain
  numbers). Rows go to a tempfile removed after the run. Flags mint, socket,
  layer_one, fetch, ue5_launch all False. JuniorSolana and the profile repo
  are untouched.

How to run
  python3 scripts/hwsw_vm_overcommit_swap_zswap_bitnetd_mmap_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_vm_overcommit_swap_zswap_bitnetd_mmap_admit_seam.py demo   # one line per fixture
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8765"
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
RANK = {"admit": 0, "soft": 1, "hold": 2, "reject": 3}


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


def num(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def evaluate(fx: dict) -> dict:
    checks, state = {}, {"worst": "admit", "flag": None}

    def mark(name, st, flag=None):
        checks[name] = st
        if RANK[st] > RANK[state["worst"]] or (st != "admit" and RANK[st] == RANK[state["worst"]] and state["flag"] is None):
            state["worst"], state["flag"] = (st, flag) if RANK[st] > RANK[state["worst"]] else (state["worst"], flag)

    cls = fx.get("cls")
    gguf, ram, swap = fx.get("gguf_mib", 0), fx.get("ram_mib", 0), fx.get("swap_mib", 0)
    vm = fx.get("vm", {})
    mark("bind", "admit" if fx.get("bind") == BIND_OK else "reject", None if fx.get("bind") == BIND_OK else "bind_reject")
    mark("cls", "admit" if cls in CLASSES else "hold", None if cls in CLASSES else "bad_class")
    if gguf <= 0:
        mark("gguf", "hold", "no_gguf")
    else:
        mark("gguf", "admit")
    oc = num(vm.get("overcommit_memory"))
    if oc is None or oc not in (0, 1, 2):
        mark("overcommit", "hold", "unparsed")
    elif oc == 1:
        mark("overcommit", "soft", "overcommit_always")
    elif oc == 0:
        ok = gguf <= ram + swap
        mark("overcommit", "admit" if ok else "hold", None if ok else "heuristic_refuse")
    else:
        ratio = num(vm.get("overcommit_ratio"))
        if ratio is None or not 0 <= ratio <= 100:
            mark("overcommit", "hold", "unparsed")
        else:
            limit = swap + ram * ratio // 100
            ok = gguf <= limit
            mark("overcommit", "admit" if ok else "hold", None if ok else "commit_limit")
    sw = num(vm.get("swappiness"))
    if sw is None or not 0 <= sw <= 200:
        mark("swappiness", "hold", "unparsed")
    elif sw > 60 and gguf * 2 > ram:
        mark("swappiness", "soft", "swappy_weights")
    else:
        mark("swappiness", "admit")
    z = str(fx.get("zswap", "N")).strip()
    if z in ("N", "0", ""):
        mark("zswap", "admit")
    elif z in ("Y", "1"):
        mark("zswap", "soft" if cls == "sbc" else "admit", "zswap_cpu_cost" if cls == "sbc" else None)
    else:
        mark("zswap", "hold", "unparsed")
    dec = state["worst"]
    order = ("bind", "cls", "gguf", "overcommit", "swappiness", "zswap")
    return {"decision": dec, "flag": state["flag"], "checks": checks,
            "codes": [CODE[checks[k]] for k in order] + [CODE[dec]]}


def fixtures():
    base = {"cls": "sff", "bind": BIND_OK, "gguf_mib": 1200, "ram_mib": 16384, "swap_mib": 4096,
            "vm": {"overcommit_memory": "0", "overcommit_ratio": "50", "swappiness": "60"}, "zswap": "N"}

    def mod(name, want, flag, **kw):
        fx = json.loads(json.dumps(base))
        for k, v in kw.items():
            if k == "vm":
                fx["vm"].update(v)
            else:
                fx[k] = v
        return name, fx, want, flag

    return [
        mod("sff_clean", "admit", None),
        mod("bad_bind", "reject", "bind_reject", bind="0.0.0.0:8765"),
        mod("bad_class", "hold", "bad_class", cls="tablet"),
        mod("no_gguf", "hold", "no_gguf", gguf_mib=0),
        mod("strict_over_limit", "hold", "commit_limit", gguf_mib=5000, ram_mib=4096, swap_mib=0,
            vm={"overcommit_memory": "2", "overcommit_ratio": "50"}),
        mod("strict_fits", "admit", None, vm={"overcommit_memory": "2", "overcommit_ratio": "50"}),
        mod("always_soft", "soft", "overcommit_always", vm={"overcommit_memory": "1"}),
        mod("heuristic_refuse", "hold", "heuristic_refuse", gguf_mib=30000),
        mod("unparsed_oc", "hold", "unparsed", vm={"overcommit_memory": "x"}),
        mod("swappy", "soft", "swappy_weights", gguf_mib=9000, vm={"swappiness": "100"}),
        mod("swappiness_range", "hold", "unparsed", vm={"swappiness": "400"}),
        mod("sbc_zswap", "soft", "zswap_cpu_cost", cls="sbc", zswap="Y"),
        mod("sff_zswap", "admit", None, zswap="1"),
        mod("zswap_junk", "hold", "unparsed", zswap="maybe"),
    ]


def run(verbose: bool = False) -> dict:
    fd, path = tempfile.mkstemp(prefix="hwsw_vm_", suffix=".jsonl")
    os.close(fd)
    prev, rows, fails = GENESIS, [], []
    try:
        with open(path, "w") as fh:
            for name, fx, want, flag in fixtures():
                r = evaluate(fx)
                if r["decision"] != want or r["flag"] != flag:
                    fails.append({"fixture": name, "got": [r["decision"], r["flag"]], "want": [want, flag]})
                text = json.dumps({"n": name, "d": r["decision"], "f": r["flag"]}, sort_keys=True)
                prev = close(prev, pack2(r["codes"]), text)
                fh.write(json.dumps({"n": name, "close": prev}) + "\n")
                rows.append((name, r["decision"], r["flag"], prev))
                if verbose:
                    print(name, r["decision"], r["flag"], prev)
    finally:
        os.remove(path)
    # determinism: second chain must match
    again, p2 = GENESIS, GENESIS
    for name, fx, _, _ in fixtures():
        r = evaluate(fx)
        again = close(again, pack2(r["codes"]), json.dumps({"n": name, "d": r["decision"], "f": r["flag"]}, sort_keys=True))
    if again != prev:
        fails.append({"fixture": "chain", "got": again, "want": prev})
    return {"ok": not fails, "rows": len(rows), "passed": len(rows) - len(fails), "tip": prev,
            "rate": RATE, "flags": FLAGS, "fails": fails}


if __name__ == "__main__":
    res = run(verbose=len(sys.argv) > 1 and sys.argv[1] == "demo")
    print(json.dumps(res, sort_keys=True))
    sys.exit(0 if res["ok"] else 1)
