#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/pressure (PSI) text -> i2sd admit plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the kernel's pressure-stall text
  (/proc/pressure/cpu, /memory, /io) tells i2sd whether the host has headroom
  to start or keep serving. Hardware classes: SBC arm64 (<=12 W, JuniorOS T4)
  and SFF host (JuniorHome T0); the SBC has the tighter limits. Distinct from
  the cgroup, hwmon, thermal, meminfo, and net-table seams (those read other
  files). Not T13 (overnight owns it). Not item 2, 4, or 5.

What it emulates
  In-memory fixtures shaped like PSI text, for example
    some avg10=1.50 avg60=0.80 avg300=0.20 total=123456
    full avg10=0.00 avg60=0.00 avg300=0.00 total=0
  (cpu has only a "some" line on older kernels). Rules, in order:
  1. Class must be "sbc" or "sff"; else hold (bad_class).
  2. Every file must parse (some/full lines, avg10/avg60/avg300 floats,
     total int); else hold (unparsed).
  3. memory full avg10 above the class limit (sbc 5.0, sff 10.0) -> reject
     (mem_full): the box is thrashing, do not start i2sd.
  4. io full avg10 above the class limit (sbc 10.0, sff 20.0) -> hold (io_stall).
  5. cpu some avg60 above the class limit (sbc 60.0, sff 80.0) -> hold (cpu_busy).
  6. memory some avg10 above 25.0 or io some avg10 above 40.0 -> soft
     (pressure_soft): start, but with the smaller thread plan.
  Plan: admit -> "start" with threads 4 (sbc) or 8 (sff); soft -> half;
  hold or reject -> none. Each row packs decision and check states as 2-bit
  trit codes (0/1/2, four per byte, I2_S order: first code in the low bits);
  rows chain with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sockets, no device opened, no subprocess,
  no systemd calls, no network fetch, no model pull, no GGUF download. Rows go
  to a tempfile removed after the run; nothing under ~/.juniorhome. Flags
  mint, socket, layer_one, fetch, ue5_launch all False. No other team file or
  earlier hwsw_* seam is imported, read, or wrapped. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_proc_pressure_psi_i2sd_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_proc_pressure_psi_i2sd_admit_seam.py demo   # one line per fixture row
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
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
LIMITS = {"sbc": {"mem_full": 5.0, "io_full": 10.0, "cpu_some60": 60.0, "threads": 4},
          "sff": {"mem_full": 10.0, "io_full": 20.0, "cpu_some60": 80.0, "threads": 8}}
MEM_SOME_SOFT, IO_SOME_SOFT = 25.0, 40.0
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
RANK = {"admit": 0, "soft": 1, "hold": 2, "reject": 3}
LINE = re.compile(r"^(some|full) avg10=(\d+(?:\.\d+)?) avg60=(\d+(?:\.\d+)?) avg300=(\d+(?:\.\d+)?) total=(\d+)$")


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


def parse_psi(text):
    """Return {"some": (a10, a60, a300), "full": ...} or None when unparsable."""
    out = {}
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    for ln in lines:
        m = LINE.match(ln.strip())
        if not m or m.group(1) in out:
            return None
        out[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return out if "some" in out else None


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

    ok_cls = cls in CLASSES
    mark("cls", "admit" if ok_cls else "hold", None if ok_cls else "bad_class")
    cpu, mem, io = (parse_psi(fx.get(k)) for k in ("cpu", "memory", "io"))
    parsed = None not in (cpu, mem, io)
    mark("parse", "admit" if parsed else "hold", None if parsed else "unparsed")
    lim = LIMITS.get(cls)
    if not (parsed and ok_cls):
        for k in ("mem_full", "io_full", "cpu_busy", "soft"):
            mark(k, "admit")
    else:
        mf = mem.get("full", (0.0,))[0]
        mark("mem_full", "reject" if mf > lim["mem_full"] else "admit", "mem_full" if mf > lim["mem_full"] else None)
        iof = io.get("full", (0.0,))[0]
        mark("io_full", "hold" if iof > lim["io_full"] else "admit", "io_stall" if iof > lim["io_full"] else None)
        busy = cpu["some"][1] > lim["cpu_some60"]
        mark("cpu_busy", "hold" if busy else "admit", "cpu_busy" if busy else None)
        soft = mem["some"][0] > MEM_SOME_SOFT or io["some"][0] > IO_SOME_SOFT
        mark("soft", "soft" if soft else "admit", "pressure_soft" if soft else None)
    threads = 0
    if worst == "admit" and lim:
        threads = lim["threads"]
    elif worst == "soft" and lim:
        threads = lim["threads"] // 2
    trits = [DECISION_CODE[worst]] + [DECISION_CODE[checks[k]] for k in
                                      ("cls", "parse", "mem_full", "io_full", "cpu_busy", "soft")]
    return {"name": fx["name"], "cls": cls, "decision": worst, "admit": worst == "admit",
            "hold": flag, "plan": "start" if threads else "none", "threads": threads,
            "checks": checks, "trits": trits}


def psi(some, full=None):
    s = "some avg10=%.2f avg60=%.2f avg300=%.2f total=%d\n" % (some[0], some[1], some[2], 1000)
    if full is not None:
        s += "full avg10=%.2f avg60=%.2f avg300=%.2f total=%d\n" % (full[0], full[1], full[2], 100)
    return s


Q = (0.0, 0.0, 0.0)
CPU_OK, MEM_OK, IO_OK = psi((1.5, 0.8, 0.2)), psi((0.1, 0.1, 0.0), Q), psi((0.5, 0.3, 0.1), Q)

FIXTURES = [
    {"name": "sff_quiet_start", "cls": "sff", "cpu": CPU_OK, "memory": MEM_OK, "io": IO_OK,
     "want": {"decision": "admit", "plan": "start", "threads": 8, "hold": None}},
    {"name": "sbc_quiet_start", "cls": "sbc", "cpu": CPU_OK, "memory": MEM_OK, "io": IO_OK,
     "want": {"decision": "admit", "threads": 4}},
    {"name": "sbc_mem_full_reject", "cls": "sbc", "cpu": CPU_OK, "memory": psi((30, 20, 10), (5.01, 3, 1)), "io": IO_OK,
     "want": {"decision": "reject", "hold": "mem_full", "plan": "none", "admit": False}},
    {"name": "sff_mem_full_same_value_ok", "cls": "sff", "cpu": CPU_OK, "memory": psi((5, 4, 3), (5.01, 3, 1)), "io": IO_OK,
     "want": {"decision": "admit", "threads": 8}},
    {"name": "sff_mem_full_at_limit_ok", "cls": "sff", "cpu": CPU_OK, "memory": psi((5, 4, 3), (10.0, 3, 1)), "io": IO_OK,
     "want": {"decision": "admit"}},
    {"name": "sff_mem_full_over_reject", "cls": "sff", "cpu": CPU_OK, "memory": psi((5, 4, 3), (10.01, 3, 1)), "io": IO_OK,
     "want": {"decision": "reject", "hold": "mem_full"}},
    {"name": "sbc_io_stall_hold", "cls": "sbc", "cpu": CPU_OK, "memory": MEM_OK, "io": psi((20, 10, 5), (10.5, 5, 2)),
     "want": {"decision": "hold", "hold": "io_stall", "plan": "none"}},
    {"name": "sff_io_full_15_ok", "cls": "sff", "cpu": CPU_OK, "memory": MEM_OK, "io": psi((20, 10, 5), (15, 5, 2)),
     "want": {"decision": "admit"}},
    {"name": "sbc_cpu_busy_hold", "cls": "sbc", "cpu": psi((90, 61, 40)), "memory": MEM_OK, "io": IO_OK,
     "want": {"decision": "hold", "hold": "cpu_busy"}},
    {"name": "sff_cpu_70_ok", "cls": "sff", "cpu": psi((90, 70, 40)), "memory": MEM_OK, "io": IO_OK,
     "want": {"decision": "admit"}},
    {"name": "sff_mem_some_soft_half", "cls": "sff", "cpu": CPU_OK, "memory": psi((26, 10, 5), (1, 1, 1)), "io": IO_OK,
     "want": {"decision": "soft", "hold": "pressure_soft", "plan": "start", "threads": 4}},
    {"name": "sbc_io_some_soft_half", "cls": "sbc", "cpu": CPU_OK, "memory": MEM_OK, "io": psi((41, 20, 10), (2, 1, 1)),
     "want": {"decision": "soft", "threads": 2}},
    {"name": "sbc_cpu_only_some_line_ok", "cls": "sbc", "cpu": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
     "memory": MEM_OK, "io": IO_OK, "want": {"decision": "admit"}},
    {"name": "sff_missing_some_hold", "cls": "sff", "cpu": "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
     "memory": MEM_OK, "io": IO_OK, "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sbc_garbage_hold", "cls": "sbc", "cpu": CPU_OK, "memory": "garbage\n", "io": IO_OK,
     "want": {"decision": "hold", "hold": "unparsed", "plan": "none"}},
    {"name": "sff_duplicate_line_hold", "cls": "sff", "cpu": CPU_OK + CPU_OK, "memory": MEM_OK, "io": IO_OK,
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sff_empty_io_hold", "cls": "sff", "cpu": CPU_OK, "memory": MEM_OK, "io": "",
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "bad_class_hold", "cls": "laptop", "cpu": CPU_OK, "memory": MEM_OK, "io": IO_OK,
     "want": {"decision": "hold", "hold": "bad_class", "plan": "none"}},
]


def run(verbose: bool = False) -> dict:
    prev, rows, fails = GENESIS, [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_psi_", suffix=".jsonl")
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
                    print(f"{r['name']:<30} decision={r['decision']:<7} hold={r['hold']!s:<14} "
                          f"threads={r['threads']} close={prev}")
    finally:
        os.remove(path)
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    return {"seam": "proc_pressure_psi->i2sd_admit",
            "roadmap": "ROADMAP #3 i2sd userspace 127.0.0.1:8767 no docker.sock",
            "rate": RATE, "flags": FLAGS, "rows": len(rows), "pack_order_ok": pack_ok, "fail": fails,
            "pass": not fails and pack_ok and not any(FLAGS.values()), "tip_close": prev}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")} if demo else out, indent=2))
    sys.exit(0 if out["pass"] else 1)
