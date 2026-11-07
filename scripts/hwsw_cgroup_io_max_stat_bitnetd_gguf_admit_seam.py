#!/usr/bin/env python3
"""Hardware-to-software seam emulator: cgroup v2 io.max + io.stat + io.pressure -> bitnetd GGUF I/O admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 2, "GGUF — operator sets JUNIOR_GGUF. Bots
  do not download.", at the seam where an operator-set GGUF meets the
  bitnetd.service cgroup I/O ceiling (io.max) and live I/O pressure
  (io.stat / io.pressure), not the memory.high ballast path and not the
  memory.max + meminfo hard GGUF/KV path. Hardware classes: SBC arm64
  (<=12 W, JuniorOS T4) and SFF host (JuniorHome T0) running bitnetd on
  127.0.0.1:8765. The seam answers: given io.max rbps/riops for a block
  device, io.stat counters, and io.pressure, may bitnetd cold-open the
  declared operator GGUF within the class open window, and at what
  soft/hold? Distinct from hwsw_cgroup_memory_high_ballast_bitnetd_admit
  (memory.high ballast) and hwsw_meminfo_cgroup_bitnetd_gguf_admit
  (memory.max + MemAvailable). This file never reads memory.* or
  /proc/meminfo. Not ROADMAP T13 (overnight owns it). Not item 3 (no
  i2sd :8767). Not item 4 (no Asahi/MLX). Not item 5 (ue5_launch stays
  False). Not a re-derivation of any earlier seam file: no other team
  file is imported or read.

What it emulates
  In-memory fixtures shaped like cgroup v2 text under
  /sys/fs/cgroup/system.slice/bitnetd.service/:
    io.max       ("maj:min rbps=.. wbps=.. riops=.. wiops=.." lines;
                  empty / missing = uncapped; MAX = uncapped key)
    io.stat      ("maj:min rbytes=.. wbytes=.. rios=.. wios=.. ...")
    io.pressure  ("some avg10=.. avg60=.. avg300=.. total=..")
  plus a declared majmin (e.g. "8:0"), gguf_bytes (JUNIOR_GGUF size;
  file never opened), the hardware class, and the bitnetd bind string.
  Rules, in order:
  1. Bind host:port must be exactly 127.0.0.1:8765; any other host
     (including 0.0.0.0) -> reject (flag bind_reject).
  2. Class must be "sbc" or "sff"; otherwise -> hold (flag bad_class).
  3. gguf_bytes must parse as a positive int; else -> hold (flag bad_gguf).
  4. Parse io.max for majmin. Unparsed device line -> hold
     (flag unparsed_max). A key set to 0 (rbps or riops) -> hold
     (flag io_zero). Missing device line / empty io.max = uncapped.
  5. Need rbps = ceil(gguf_bytes / open_window_s) with SBC window 30 s,
     SFF window 10 s. Floor also enforces class_min_rbps (SBC 8 MiB/s,
     SFF 64 MiB/s) and class_min_riops (SBC 64, SFF 512) when the key
     is capped. If capped rbps < need -> hold (flag rbps_floor). If
     capped riops < class_min_riops -> hold (flag riops_floor).
  6. Parse io.stat rbytes for majmin (informational; unparsed/missing
     treated as 0, fail-open on counters only).
  7. Parse io.pressure some avg10. Unparsed or missing -> 0.0.
     avg10 >= 20.0 -> hold (flag pressure_hard). avg10 >= 10.0 -> soft
     (flag pressure_soft; admit stays false, no auto-fetch).
  8. Admit only when bind ok, I/O floors met, and pressure is not
     soft/hold.
  Each row packs decision and check states as real 2-bit trit codes
  (0/1/2, four per byte, I2_S order: first code in the low bits), and rows
  are chained with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no cgroup writes, no memory.high /
  memory.max / memory.current / memory.pressure, no /proc/meminfo, no
  systemd calls, no file opened at JUNIOR_GGUF, no block device opened,
  no io_uring / pread, no model pull, no GGUF download, no sockets
  (binds are strings; nothing listens or connects), no device opened,
  no network fetch, no subprocess. It does not import or re-run any
  team file or any earlier hwsw_* seam (including memory.high ballast,
  memory.max GGUF admit, cgroup cpu.max i2sd, net-class i2sd, DRM EDID
  Deck, thermal/cpufreq, block mount placement). Rows go to a tempfile
  removed after the run; nothing under ~/.juniorhome. Flags: mint,
  socket, layer_one, fetch, ue5_launch all False. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_cgroup_io_max_stat_bitnetd_gguf_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_cgroup_io_max_stat_bitnetd_gguf_admit_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8765"
MIB = 1024 * 1024
# Cold-open window: how long bitnetd may take to sequential-read the GGUF.
OPEN_WINDOW_S = {"sbc": 30.0, "sff": 10.0}
# Class floors applied when the matching io.max key is capped (not MAX).
MIN_RBPS = {"sbc": 8 * MIB, "sff": 64 * MIB}
MIN_RIOPS = {"sbc": 64, "sff": 512}
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


def _parse_io_kv(toks: list) -> dict:
    """Parse rbps=/wbps=/riops=/wiops=/rbytes=/... tokens; MAX -> None (uncapped)."""
    out = {}
    for tok in toks:
        if "=" not in tok:
            continue
        k, _, v = tok.partition("=")
        k = k.strip().lower()
        v = v.strip()
        if not k:
            continue
        if v.upper() == "MAX":
            out[k] = None
            continue
        n = num(v)
        if n is None or n < 0:
            out[k] = "BAD"
            continue
        out[k] = n
    return out


def parse_io_max(raw, majmin: str) -> tuple:
    """Return (limits_dict_or_None_uncapped, reason_or_None).

    limits keys: rbps, riops (None value = uncapped key). Empty/missing
    device line means fully uncapped (None, None).
    """
    s = str(raw).strip() if raw is not None else ""
    if not s:
        return None, None
    found = False
    for line in s.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] != majmin:
            continue
        found = True
        kv = _parse_io_kv(parts[1:])
        if any(v == "BAD" for v in kv.values()):
            return None, "unparsed_max"
        # Only read-side keys gate GGUF cold open.
        return {"rbps": kv.get("rbps", None), "riops": kv.get("riops", None)}, None
    if not found:
        return None, None  # uncapped: device not listed
    return None, "unparsed_max"


def parse_io_stat_rbytes(raw, majmin: str) -> int:
    """Best-effort rbytes for majmin; missing/unparsed -> 0."""
    s = str(raw).strip() if raw is not None else ""
    if not s:
        return 0
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts or parts[0] != majmin:
            continue
        kv = _parse_io_kv(parts[1:])
        v = kv.get("rbytes", 0)
        if v == "BAD" or v is None:
            return 0
        return int(v)
    return 0


def parse_pressure_avg10(raw) -> float:
    """Best-effort some avg10; missing/unparsed -> 0.0."""
    s = str(raw).strip() if raw is not None else ""
    if not s:
        return 0.0
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
    majmin = str(fx.get("majmin", "8:0")).strip()
    io_max = fx.get("io_max", "")
    io_stat = fx.get("io_stat", "")
    io_pressure = fx.get("io_pressure", "")
    gguf = num(fx.get("gguf_bytes"))

    bind_ok = bind == BIND_OK
    if not bind_ok:
        return _row(fx, "reject", "bind_reject", bind_ok=False, io_ok=False,
                    need_rbps=None, rbps=None, riops=None, avg10=None, rbytes=0)

    if cls not in ("sbc", "sff"):
        return _row(fx, "hold", "bad_class", bind_ok=True, io_ok=False,
                    need_rbps=None, rbps=None, riops=None, avg10=None, rbytes=0)

    if gguf is None or gguf <= 0:
        return _row(fx, "hold", "bad_gguf", bind_ok=True, io_ok=False,
                    need_rbps=None, rbps=None, riops=None, avg10=None, rbytes=0)

    limits, max_reason = parse_io_max(io_max, majmin)
    if max_reason == "unparsed_max":
        return _row(fx, "hold", "unparsed_max", bind_ok=True, io_ok=False,
                    need_rbps=None, rbps=None, riops=None, avg10=None, rbytes=0)

    need_rbps = int(math.ceil(gguf / OPEN_WINDOW_S[cls]))
    # Also enforce class minimum throughput when calculating need.
    need_rbps = max(need_rbps, MIN_RBPS[cls])
    need_riops = MIN_RIOPS[cls]

    rbps = None if limits is None else limits.get("rbps")
    riops = None if limits is None else limits.get("riops")

    if rbps == 0 or riops == 0:
        return _row(fx, "hold", "io_zero", bind_ok=True, io_ok=False,
                    need_rbps=need_rbps, rbps=rbps, riops=riops, avg10=None, rbytes=0)

    if rbps is not None and rbps < need_rbps:
        return _row(fx, "hold", "rbps_floor", bind_ok=True, io_ok=False,
                    need_rbps=need_rbps, rbps=rbps, riops=riops, avg10=None, rbytes=0)

    if riops is not None and riops < need_riops:
        return _row(fx, "hold", "riops_floor", bind_ok=True, io_ok=False,
                    need_rbps=need_rbps, rbps=rbps, riops=riops, avg10=None, rbytes=0)

    rbytes = parse_io_stat_rbytes(io_stat, majmin)
    avg10 = parse_pressure_avg10(io_pressure)

    if avg10 >= PRESSURE_HARD:
        return _row(fx, "hold", "pressure_hard", bind_ok=True, io_ok=True,
                    need_rbps=need_rbps, rbps=rbps, riops=riops, avg10=avg10, rbytes=rbytes)

    if avg10 >= PRESSURE_SOFT:
        return _row(fx, "soft", "pressure_soft", bind_ok=True, io_ok=True,
                    need_rbps=need_rbps, rbps=rbps, riops=riops, avg10=avg10, rbytes=rbytes)

    return _row(fx, "admit", None, bind_ok=True, io_ok=True,
                need_rbps=need_rbps, rbps=rbps, riops=riops, avg10=avg10, rbytes=rbytes)


def _row(fx, decision, hold_reason, bind_ok, io_ok, need_rbps, rbps, riops, avg10, rbytes):
    trits = [
        DECISION_CODE[decision],
        2 if bind_ok else 0,
        2 if io_ok else 0,
        2 if decision == "admit" else (1 if decision == "soft" else 0),
        2 if fx.get("cls") == "sff" else (1 if fx.get("cls") == "sbc" else 0),
        2 if (avg10 is not None and avg10 < PRESSURE_SOFT) else (
            1 if (avg10 is not None and avg10 < PRESSURE_HARD) else 0),
    ]
    return {
        "name": fx["name"],
        "cls": fx.get("cls"),
        "majmin": fx.get("majmin", "8:0"),
        "bind_ok": bind_ok,
        "io_ok": io_ok,
        "need_rbps": need_rbps,
        "rbps": rbps,
        "riops": riops,
        "rbytes": rbytes,
        "avg10": avg10,
        "decision": decision,
        "hold": hold_reason,
        "admit": decision == "admit",
        "trits": trits,
    }


FIXTURES = [
    {
        "name": "sff_uncapped_cool_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "",
        "io_stat": "8:0 rbytes=1048576 wbytes=0 rios=16 wios=0 dbytes=0 dios=0",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 512 * MIB,
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_rbps_ok_admit",
        "cls": "sbc",
        "bind": BIND_OK,
        "majmin": "179:0",
        "io_max": "179:0 rbps=16777216 wbps=MAX riops=256 wiops=MAX",  # 16 MiB/s >= 8
        "io_stat": "179:0 rbytes=0 wbytes=0 rios=0 wios=0 dbytes=0 dios=0",
        "io_pressure": "some avg10=1.50 avg60=0.80 avg300=0.40 total=12",
        "gguf_bytes": 128 * MIB,  # need max(128Mi/30, 8Mi) = ~8.5Mi -> still under 16Mi
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sff_max_keys_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "majmin": "8:16",
        "io_max": "8:16 rbps=MAX wbps=MAX riops=MAX wiops=MAX",
        "io_stat": "8:16 rbytes=4096 wbytes=0 rios=1 wios=0 dbytes=0 dios=0",
        "io_pressure": "some avg10=0.10 avg60=0.10 avg300=0.10 total=1",
        "gguf_bytes": 2 * 1024 * MIB,
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_rbps_floor_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "8:0 rbps=1048576 wbps=MAX riops=256 wiops=MAX",  # 1 MiB/s < 8
        "io_stat": "",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 64 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "rbps_floor"},
    },
    {
        "name": "sff_riops_floor_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "8:0 rbps=MAX wbps=MAX riops=64 wiops=MAX",  # 64 < 512
        "io_stat": "",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 256 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "riops_floor"},
    },
    {
        "name": "sff_io_zero_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "8:0 rbps=0 wbps=MAX riops=MAX wiops=MAX",
        "io_stat": "",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 256 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "io_zero"},
    },
    {
        "name": "sbc_pressure_soft",
        "cls": "sbc",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "",
        "io_stat": "8:0 rbytes=8192 wbytes=0 rios=2 wios=0 dbytes=0 dios=0",
        "io_pressure": "some avg10=12.00 avg60=8.00 avg300=4.00 total=99",
        "gguf_bytes": 64 * MIB,
        "want": {"decision": "soft", "admit": False, "hold": "pressure_soft"},
    },
    {
        "name": "sff_pressure_hard_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "8:0 rbps=MAX wbps=MAX riops=MAX wiops=MAX",
        "io_stat": "",
        "io_pressure": "some avg10=25.00 avg60=18.00 avg300=10.00 total=500",
        "gguf_bytes": 512 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "pressure_hard"},
    },
    {
        "name": "sff_wildcard_bind_reject",
        "cls": "sff",
        "bind": "0.0.0.0:8765",
        "majmin": "8:0",
        "io_max": "",
        "io_stat": "",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 256 * MIB,
        "want": {"decision": "reject", "admit": False, "hold": "bind_reject"},
    },
    {
        "name": "sbc_unparsed_max_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "8:0 rbps=nope wiops=MAX",
        "io_stat": "",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 64 * MIB,
        "want": {"decision": "hold", "admit": False, "hold": "unparsed_max"},
    },
    {
        "name": "sff_bad_gguf_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "majmin": "8:0",
        "io_max": "",
        "io_stat": "",
        "io_pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=0",
        "gguf_bytes": 0,
        "want": {"decision": "hold", "admit": False, "hold": "bad_gguf"},
    },
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    rows = []
    fails = []
    fd, path = tempfile.mkstemp(prefix="hwsw_cgroup_io_", suffix=".jsonl")
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
                          f"hold={r['hold']!s:<14} need_rbps={r['need_rbps']!s:<12} "
                          f"avg10={r['avg10']!s:<6} close={prev}")
    finally:
        os.remove(path)
    return {
        "seam": "cgroup_io_max_stat_pressure->bitnetd_gguf_admit",
        "roadmap": "ROADMAP #2 GGUF operator set",
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
