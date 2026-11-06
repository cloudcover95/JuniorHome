#!/usr/bin/env python3
"""Hardware-to-software seam emulator: cgroup v2 memory.high + memory.current ballast -> bitnetd admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 2, "GGUF — operator sets JUNIOR_GGUF. Bots
  do not download.", at the seam where an operator-set GGUF meets the
  soft memory ceiling of the bitnetd.service cgroup (memory.high) rather
  than the hard kill limit (memory.max). Hardware classes: SBC arm64
  (<=12 W, JuniorOS T4) and SFF host (JuniorHome T0) running bitnetd on
  127.0.0.1:8765. The seam answers: given memory.high, memory.current, and
  memory.pressure, may bitnetd keep a declared anon ballast (page-cache
  stand-in for a cold GGUF working set) and still admit the operator GGUF,
  and at what soft/hold? Distinct from hwsw_meminfo_cgroup_bitnetd_gguf_admit
  (that file uses memory.max + MemAvailable for hard GGUF/KV sizing). This
  file never reads memory.max or /proc/meminfo. Not ROADMAP T13 (overnight
  owns it). Not item 4 (no Asahi/MLX). Not item 5 (ue5_launch stays False).
  Not a re-derivation of any earlier seam file: no other team file is
  imported or read.

What it emulates
  In-memory fixtures shaped like cgroup v2 text under
  /sys/fs/cgroup/system.slice/bitnetd.service/:
    memory.high     ("max" or byte count)
    memory.current  (bytes currently charged)
    memory.pressure ("some avg10=.. avg60=.. avg300=.. total=..")
  plus a declared ballast_bytes (anon stand-in; never allocated), a
  declared gguf_bytes (JUNIOR_GGUF size; file never opened), the hardware
  class, and the bitnetd bind string.
  Rules, in order:
  1. Bind host:port must be exactly 127.0.0.1:8765; any other host
     (including 0.0.0.0) -> reject (flag bind_reject).
  2. Class must be "sbc" or "sff"; otherwise -> hold (flag bad_class).
  3. Parse memory.high. Bare "max" means no soft ceiling (high=None /
     uncapped). Otherwise a non-negative int of bytes. Unparsed -> hold
     (flag unparsed_high). high == 0 -> hold (flag high_zero).
  4. Parse memory.current as non-negative int. Unparsed -> hold
     (flag unparsed_current).
  5. If high is set and current > high -> hold (flag over_high). The
     kernel would already be reclaiming; bitnetd must not grow ballast.
  6. Headroom: when high is set, headroom = high - current; when uncapped
     ("max"), headroom = class_uncapped_budget (SBC 2 GiB, SFF 8 GiB) -
     current (floor 0).
  7. Need = gguf_bytes + ballast_bytes + class_reserve (SBC 256 MiB,
     SFF 1 GiB). Need must be > 0 and gguf_bytes > 0; else hold
     (flag bad_need).
  8. If need > headroom -> hold (flag headroom).
  9. Parse memory.pressure some avg10. Unparsed or missing -> treat as
     0.0 (fail-open on pressure text only; hard fields already gated).
     avg10 >= 20.0 -> hold (flag pressure_hard). avg10 >= 10.0 -> soft
     (flag pressure_soft; admit ballast at half declared size in the
     decision text, but do not auto-fetch a smaller GGUF).
  10. Admit only when bind ok, headroom covers need, and pressure is not
      soft/hold.
  Each row packs decision and check states as real 2-bit trit codes
  (0/1/2, four per byte, I2_S order: first code in the low bits), and rows
  are chained with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no cgroup writes, no memory.max, no
  /proc/meminfo, no systemd calls, no file opened at JUNIOR_GGUF, no
  anon mmap / madvise / ballast allocation, no model pull, no GGUF
  download, no sockets (binds are strings; nothing listens or connects),
  no device opened, no network fetch, no subprocess. It does not import
  or re-run any team file or any earlier hwsw_* seam (including the
  memory.max GGUF admit, cgroup cpu.max i2sd, thermal/cpufreq, block
  mount placement, and so on). Rows go to a tempfile removed after the
  run; nothing under ~/.juniorhome. Flags: mint, socket, layer_one,
  fetch, ue5_launch all False. JuniorSolana and the profile repo are
  untouched.

How to run
  python3 scripts/hwsw_cgroup_memory_high_ballast_bitnetd_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_cgroup_memory_high_ballast_bitnetd_admit_seam.py demo   # one line per fixture row
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
MIB = 1024 * 1024
GIB = 1024 * MIB
RESERVE = {"sbc": 256 * MIB, "sff": 1 * GIB}
# When memory.high is "max", pretend a class soft budget for ballast math only.
UNCAPPED_BUDGET = {"sbc": 2 * GIB, "sff": 8 * GIB}
PRESSURE_SOFT = 10.0
PRESSURE_HARD = 20.0
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}


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


def num(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def fnum(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse_high(raw) -> tuple:
    """Return (high_bytes_or_None_for_max, reason_or_None)."""
    s = str(raw).strip() if raw is not None else ""
    if s == "max":
        return None, None
    v = num(s)
    if v is None or v < 0:
        return None, "unparsed_high"
    if v == 0:
        return 0, "high_zero"
    return v, None


def parse_current(raw) -> tuple:
    v = num(raw)
    if v is None or v < 0:
        return None, "unparsed_current"
    return v, None


def parse_pressure_avg10(raw) -> float:
    """Best-effort some avg10; missing/unparsed -> 0.0."""
    s = str(raw).strip() if raw is not None else ""
    if not s:
        return 0.0
    # Prefer the "some" line; fall back to first avg10= on any line.
    some_avg = None
    any_avg = None
    for line in s.splitlines():
        line = line.strip()
        if "avg10=" not in line:
            continue
        for tok in line.split():
            if tok.startswith("avg10="):
                val = fnum(tok.split("=", 1)[1])
                if val is None:
                    continue
                if line.startswith("some"):
                    some_avg = val
                elif any_avg is None:
                    any_avg = val
    if some_avg is not None:
        return some_avg
    if any_avg is not None:
        return any_avg
    return 0.0


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    bind = fx.get("bind", "")
    high_raw = fx.get("memory_high", "")
    current_raw = fx.get("memory_current", "")
    pressure_raw = fx.get("memory_pressure", "")
    gguf = num(fx.get("gguf_bytes"))
    ballast = num(fx.get("ballast_bytes"))

    bind_ok = bind == BIND_OK
    if not bind_ok:
        return _row(fx, "reject", "bind_reject", bind_ok=False, high_ok=False,
                    current_ok=False, headroom=None, need=None, avg10=None)

    if cls not in ("sbc", "sff"):
        return _row(fx, "hold", "bad_class", bind_ok=True, high_ok=False,
                    current_ok=False, headroom=None, need=None, avg10=None)

    high, high_reason = parse_high(high_raw)
    if high_reason == "unparsed_high":
        return _row(fx, "hold", "unparsed_high", bind_ok=True, high_ok=False,
                    current_ok=False, headroom=None, need=None, avg10=None)
    if high_reason == "high_zero":
        return _row(fx, "hold", "high_zero", bind_ok=True, high_ok=False,
                    current_ok=False, headroom=0, need=None, avg10=None)

    current, cur_reason = parse_current(current_raw)
    if cur_reason:
        return _row(fx, "hold", "unparsed_current", bind_ok=True, high_ok=True,
                    current_ok=False, headroom=None, need=None, avg10=None)

    if high is not None and current > high:
        return _row(fx, "hold", "over_high", bind_ok=True, high_ok=True,
                    current_ok=True, headroom=0, need=None, avg10=None)

    if high is None:
        budget = UNCAPPED_BUDGET[cls]
        headroom = max(0, budget - current)
    else:
        headroom = high - current

    if gguf is None or ballast is None or gguf <= 0 or ballast < 0:
        return _row(fx, "hold", "bad_need", bind_ok=True, high_ok=True,
                    current_ok=True, headroom=headroom, need=None, avg10=None)

    need = gguf + ballast + RESERVE[cls]
    if need > headroom:
        return _row(fx, "hold", "headroom", bind_ok=True, high_ok=True,
                    current_ok=True, headroom=headroom, need=need, avg10=None)

    avg10 = parse_pressure_avg10(pressure_raw)
    if avg10 >= PRESSURE_HARD:
        return _row(fx, "hold", "pressure_hard", bind_ok=True, high_ok=True,
                    current_ok=True, headroom=headroom, need=need, avg10=avg10)
    if avg10 >= PRESSURE_SOFT:
        return _row(fx, "soft", "pressure_soft", bind_ok=True, high_ok=True,
                    current_ok=True, headroom=headroom, need=need, avg10=avg10,
                    ballast_soft=ballast // 2)

    return _row(fx, "admit", None, bind_ok=True, high_ok=True,
                current_ok=True, headroom=headroom, need=need, avg10=avg10)


def _row(fx, decision, hold_reason, bind_ok, high_ok, current_ok, headroom, need, avg10,
         ballast_soft=None):
    trits = [
        DECISION_CODE[decision],
        2 if bind_ok else 0,
        2 if high_ok else 0,
        2 if current_ok else 0,
        2 if fx.get("cls") == "sff" else (1 if fx.get("cls") == "sbc" else 0),
        2 if decision == "admit" else (1 if decision == "soft" else 0),
    ]
    return {
        "name": fx["name"],
        "cls": fx.get("cls"),
        "bind_ok": bind_ok,
        "high_ok": high_ok,
        "current_ok": current_ok,
        "headroom": headroom,
        "need": need,
        "avg10": avg10,
        "ballast_soft": ballast_soft,
        "decision": decision,
        "hold": hold_reason,
        "admit": decision == "admit",
        "trits": trits,
    }


FIXTURES = [
    {
        "name": "sff_high_room_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "memory_high": str(12 * GIB),
        "memory_current": str(2 * GIB),
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 4 * GIB,
        "ballast_bytes": 2 * GIB,
        # need = 4G + 2G + 1G = 7G; headroom = 10G
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_high_room_admit",
        "cls": "sbc",
        "bind": BIND_OK,
        "memory_high": str(3 * GIB),
        "memory_current": str(512 * MIB),
        "memory_pressure": "some avg10=1.00 avg60=0.50 avg300=0.20 total=100",
        "gguf_bytes": 900 * MIB,
        "ballast_bytes": 400 * MIB,
        # need = 900M + 400M + 256M = 1556M; headroom = 2560M
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sff_uncapped_max_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "memory_high": "max",
        "memory_current": str(1 * GIB),
        "memory_pressure": "some avg10=0.10 avg60=0.10 avg300=0.10 total=1",
        "gguf_bytes": 3 * GIB,
        "ballast_bytes": 1 * GIB,
        # uncapped budget 8G - 1G = 7G; need = 3+1+1 = 5G
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_headroom_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "memory_high": str(2 * GIB),
        "memory_current": str(1500 * MIB),
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 900 * MIB,
        "ballast_bytes": 400 * MIB,
        # headroom = 548M; need = 1556M
        "want": {"decision": "hold", "admit": False, "hold": "headroom"},
    },
    {
        "name": "sff_over_high_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "memory_high": str(4 * GIB),
        "memory_current": str(5 * GIB),
        "memory_pressure": "some avg10=5.00 avg60=4.00 avg300=3.00 total=50",
        "gguf_bytes": 1 * GIB,
        "ballast_bytes": 512 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "over_high"},
    },
    {
        "name": "sbc_pressure_soft",
        "cls": "sbc",
        "bind": BIND_OK,
        "memory_high": str(4 * GIB),
        "memory_current": str(512 * MIB),
        "memory_pressure": "some avg10=12.50 avg60=8.00 avg300=4.00 total=2000",
        "gguf_bytes": 900 * MIB,
        "ballast_bytes": 400 * MIB,
        "want": {"decision": "soft", "admit": False, "hold": "pressure_soft"},
    },
    {
        "name": "sff_pressure_hard_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "memory_high": str(16 * GIB),
        "memory_current": str(2 * GIB),
        "memory_pressure": "some avg10=25.00 avg60=18.00 avg300=10.00 total=9000",
        "gguf_bytes": 4 * GIB,
        "ballast_bytes": 2 * GIB,
        "want": {"decision": "hold", "admit": False, "hold": "pressure_hard"},
    },
    {
        "name": "sff_wildcard_bind_reject",
        "cls": "sff",
        "bind": "0.0.0.0:8765",
        "memory_high": "max",
        "memory_current": str(1 * GIB),
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 1 * GIB,
        "ballast_bytes": 512 * MIB,
        "want": {"decision": "reject", "admit": False, "hold": "bind_reject"},
    },
    {
        "name": "sbc_high_zero_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "memory_high": "0",
        "memory_current": "0",
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 100 * MIB,
        "ballast_bytes": 50 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "high_zero"},
    },
    {
        "name": "sff_high_unparsed_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "memory_high": "unlimited",
        "memory_current": str(1 * GIB),
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 1 * GIB,
        "ballast_bytes": 512 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "unparsed_high"},
    },
    {
        "name": "sbc_current_unparsed_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "memory_high": str(2 * GIB),
        "memory_current": "n/a",
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 100 * MIB,
        "ballast_bytes": 50 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "unparsed_current"},
    },
    {
        "name": "sff_bad_need_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "memory_high": str(8 * GIB),
        "memory_current": str(1 * GIB),
        "memory_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 0,
        "ballast_bytes": 512 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "bad_need"},
    },
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    rows = []
    fails = []
    fd, path = tempfile.mkstemp(prefix="hwsw_cgroup_memhigh_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for fx in FIXTURES:
                r = evaluate(fx)
                text = json.dumps({k: v for k, v in r.items() if k != "trits"}, sort_keys=True)
                prev = close(prev, pack2(r["trits"]), text)
                r["close"] = prev
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                ok = all(r.get(k) == v for k, v in fx["want"].items())
                if not ok:
                    fails.append({"name": fx["name"], "got": {k: r.get(k) for k in fx["want"]},
                                  "want": fx["want"]})
                rows.append(r)
                if verbose:
                    print(f"{r['name']:<28} decision={r['decision']!s:<7} "
                          f"hold={r['hold']!s:<16} headroom={r['headroom']!s:<12} "
                          f"need={r['need']!s:<12} close={prev}")
    finally:
        os.remove(path)
    return {
        "seam": "cgroup_memory_high_ballast->bitnetd_admit",
        "roadmap": "ROADMAP #2 GGUF",
        "rate": RATE,
        "bind": BIND_OK,
        "flags": FLAGS,
        "rows": len(rows),
        "fail": [f["name"] if isinstance(f, dict) else f for f in fails],
        "pass": not fails,
        "tip_close": prev,
    }


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    if not demo:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")}, indent=2))
    sys.exit(0 if out["pass"] else 1)
