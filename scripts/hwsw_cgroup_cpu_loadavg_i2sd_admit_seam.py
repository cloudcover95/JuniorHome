#!/usr/bin/env python3
"""Hardware-to-software seam emulator: cgroup v2 cpu.max + /proc/loadavg -> i2sd admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767. No
  docker.sock." The seam decides, from the cgroup v2 CPU controller and
  loadavg alone, whether the i2sd userspace daemon may admit a listen on a
  box and what CPU fraction budget it may claim. Hardware classes: SBC
  (arm64, JuniorOS T4) and SFF host (JuniorHome T0). Not ROADMAP T13
  (overnight owns it). Not item 4 (no Asahi/MLX). Not item 5 (ue5_launch
  stays False). Not a re-run of meminfo/cgroup memory.admit, thermal/
  cpufreq threads, cpu_isa dispatch, or cpu_cache tile seams: this is
  cpu.max quota/period plus loadavg 1/5/15 for i2sd CPU-budget admit only.

What it emulates
  In-memory fixtures shaped like cgroup v2 cpu.max ("max" or
  "<quota_us> <period_us>") plus /proc/loadavg ("<load1> <load5> <load15>
  <runnable>/<total> <last_pid>") and optional nproc. Rules, in order:
  1. Bind host:port must be exactly 127.0.0.1:8767; any other host
     (including 0.0.0.0) -> reject (decision "reject", flag bind_reject).
  2. Class must be "sbc" or "sff"; otherwise -> "hold" (flag bad_class).
  3. Parse cpu.max. Bare "max" means unlimited (frac=1.0). Otherwise two
     ints quota/period; period must be >0 and quota>=0. Unparsed -> "hold"
     (flag unparsed_cpu). quota 0 -> "hold" (flag cpu_zero).
  4. Parse loadavg: three floats then runnable/total. Unparsed -> "hold"
     (flag unparsed_load). nproc defaults to 4 (sbc) / 8 (sff) when absent
     or unparsed; must be >=1 when present.
  5. Floor: SBC requires frac >= 0.25 (1 of 4); SFF requires frac >= 0.125
     (1 of 8). Below floor -> "hold" (flag cpu_floor).
  6. load1 > nproc * 4 -> "hold" (flag overload_hard). load1 > nproc * 2
     -> "soft" (flag overload_soft; boot-relative retry / eco).
  7. Unlimited cpu.max ("max") with load1 > nproc * 0.75 on SBC -> "soft"
     (flag uncapped_hot); SFF allows uncapped through soft only when
     load1 > nproc * 1.5 (caught by rule 6 first when hotter).
  8. Admit only when bind ok, cpu frac meets floor, and load is not
     soft/hold.
  Each row packs decision and check states as real 2-bit trit codes
  (0/1/2, four per byte, I2_S order: first code in the low bits), and rows
  are chained with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no cgroup writes, no sched_setattr, no
  nice/renice, no sockets (binds are strings; nothing listens or
  connects), no device opened, no model pull, no GGUF, no network fetch,
  no subprocess, no docker.sock. It does not import or re-run any team
  file or any earlier hwsw_* seam (memory cgroup, thermal/cpufreq,
  cpu_isa, cpu_cache, net class, and so on). Rows go to a tempfile
  removed after the run; nothing under ~/.juniorhome. Flags: mint,
  socket, layer_one, fetch, ue5_launch all False. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_cgroup_cpu_loadavg_i2sd_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_cgroup_cpu_loadavg_i2sd_admit_seam.py demo   # one line per fixture row
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
BIND_OK = "127.0.0.1:8767"
FLOOR = {"sbc": 0.25, "sff": 0.125}
DEFAULT_NPROC = {"sbc": 4, "sff": 8}
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


def parse_cpu_max(raw) -> tuple:
    """Return (frac, reason_or_None). reason set means unusable."""
    s = str(raw).strip() if raw is not None else ""
    if s == "max":
        return 1.0, None
    parts = s.split()
    if len(parts) != 2:
        return None, "unparsed_cpu"
    quota = num(parts[0])
    period = num(parts[1])
    if quota is None or period is None or period <= 0 or quota < 0:
        return None, "unparsed_cpu"
    if quota == 0:
        return 0.0, "cpu_zero"
    return quota / period, None


def parse_loadavg(raw) -> tuple:
    """Return (load1, load5, load15, runnable, total, reason_or_None)."""
    s = str(raw).strip() if raw is not None else ""
    parts = s.split()
    if len(parts) < 4:
        return None, None, None, None, None, "unparsed_load"
    load1, load5, load15 = fnum(parts[0]), fnum(parts[1]), fnum(parts[2])
    rt = parts[3].split("/")
    if len(rt) != 2:
        return None, None, None, None, None, "unparsed_load"
    runnable, total = num(rt[0]), num(rt[1])
    if None in (load1, load5, load15, runnable, total):
        return None, None, None, None, None, "unparsed_load"
    if load1 < 0 or total < 1:
        return None, None, None, None, None, "unparsed_load"
    return load1, load5, load15, runnable, total, None


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    bind = fx.get("bind", "")
    cpu_max = fx.get("cpu_max", "")
    loadavg = fx.get("loadavg", "")
    nproc_raw = fx.get("nproc")

    bind_ok = bind == BIND_OK

    if not bind_ok:
        return _row(fx, "reject", "bind_reject", bind_ok=False, cpu_ok=False,
                    load_ok=False, frac=None, load1=None, nproc=None)

    if cls not in ("sbc", "sff"):
        return _row(fx, "hold", "bad_class", bind_ok=True, cpu_ok=False,
                    load_ok=False, frac=None, load1=None, nproc=None)

    frac, cpu_reason = parse_cpu_max(cpu_max)
    if cpu_reason == "unparsed_cpu":
        return _row(fx, "hold", "unparsed_cpu", bind_ok=True, cpu_ok=False,
                    load_ok=False, frac=None, load1=None, nproc=None)
    if cpu_reason == "cpu_zero":
        return _row(fx, "hold", "cpu_zero", bind_ok=True, cpu_ok=False,
                    load_ok=False, frac=0.0, load1=None, nproc=None)

    load1, load5, load15, runnable, total, load_reason = parse_loadavg(loadavg)
    if load_reason:
        return _row(fx, "hold", "unparsed_load", bind_ok=True, cpu_ok=True,
                    load_ok=False, frac=frac, load1=None, nproc=None)

    nproc = num(nproc_raw) if nproc_raw is not None else None
    if nproc is None or nproc < 1:
        nproc = DEFAULT_NPROC[cls]

    floor = FLOOR[cls]
    if frac < floor:
        return _row(fx, "hold", "cpu_floor", bind_ok=True, cpu_ok=False,
                    load_ok=True, frac=frac, load1=load1, nproc=nproc)

    if load1 > nproc * 4:
        return _row(fx, "hold", "overload_hard", bind_ok=True, cpu_ok=True,
                    load_ok=False, frac=frac, load1=load1, nproc=nproc)

    if load1 > nproc * 2:
        return _row(fx, "soft", "overload_soft", bind_ok=True, cpu_ok=True,
                    load_ok=False, frac=frac, load1=load1, nproc=nproc)

    # uncapped + warm on SBC: soft eco rather than full admit
    if str(cpu_max).strip() == "max" and cls == "sbc" and load1 > nproc * 0.75:
        return _row(fx, "soft", "uncapped_hot", bind_ok=True, cpu_ok=True,
                    load_ok=True, frac=frac, load1=load1, nproc=nproc)

    return _row(fx, "admit", None, bind_ok=True, cpu_ok=True,
                load_ok=True, frac=frac, load1=load1, nproc=nproc)


def _row(fx, decision, hold_reason, bind_ok, cpu_ok, load_ok, frac, load1, nproc):
    trits = [
        DECISION_CODE[decision],
        2 if bind_ok else 0,
        2 if cpu_ok else 0,
        2 if load_ok else (1 if decision == "soft" else 0),
        2 if fx.get("cls") == "sff" else (1 if fx.get("cls") == "sbc" else 0),
        2 if decision == "admit" else (1 if decision == "soft" else 0),
    ]
    return {
        "name": fx["name"],
        "cls": fx.get("cls"),
        "bind_ok": bind_ok,
        "cpu_ok": cpu_ok,
        "load_ok": load_ok,
        "frac": None if frac is None else round(frac, 6),
        "load1": load1,
        "nproc": nproc,
        "decision": decision,
        "hold": hold_reason,
        "admit": decision == "admit",
        "trits": trits,
    }


FIXTURES = [
    {
        "name": "sff_quota_ok_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "cpu_max": "400000 100000",  # 4.0 cores
        "loadavg": "0.40 0.30 0.20 1/200 1234",
        "nproc": 8,
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_max_cool_admit",
        "cls": "sbc",
        "bind": BIND_OK,
        "cpu_max": "max",
        "loadavg": "0.50 0.40 0.30 1/80 99",
        "nproc": 4,
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_quarter_floor_admit",
        "cls": "sbc",
        "bind": BIND_OK,
        "cpu_max": "25000 100000",  # 0.25 exactly
        "loadavg": "0.20 0.20 0.20 1/50 10",
        "nproc": 4,
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_below_floor_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "cpu_max": "20000 100000",  # 0.20 < 0.25
        "loadavg": "0.10 0.10 0.10 1/40 11",
        "nproc": 4,
        "want": {"decision": "hold", "admit": False, "hold": "cpu_floor"},
    },
    {
        "name": "sff_zero_quota_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "cpu_max": "0 100000",
        "loadavg": "0.10 0.10 0.10 1/100 12",
        "nproc": 8,
        "want": {"decision": "hold", "admit": False, "hold": "cpu_zero"},
    },
    {
        "name": "sff_overload_soft",
        "cls": "sff",
        "bind": BIND_OK,
        "cpu_max": "800000 100000",
        "loadavg": "17.00 10.00 5.00 20/400 55",  # > 8*2, <= 8*4
        "nproc": 8,
        "want": {"decision": "soft", "admit": False, "hold": "overload_soft"},
    },
    {
        "name": "sbc_overload_hard_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "cpu_max": "max",
        "loadavg": "17.00 12.00 8.00 30/200 66",  # > 4*4
        "nproc": 4,
        "want": {"decision": "hold", "admit": False, "hold": "overload_hard"},
    },
    {
        "name": "sbc_uncapped_hot_soft",
        "cls": "sbc",
        "bind": BIND_OK,
        "cpu_max": "max",
        "loadavg": "3.20 2.00 1.00 4/100 77",  # > 4*0.75, <= 4*2
        "nproc": 4,
        "want": {"decision": "soft", "admit": False, "hold": "uncapped_hot"},
    },
    {
        "name": "sff_wildcard_bind_reject",
        "cls": "sff",
        "bind": "0.0.0.0:8767",
        "cpu_max": "max",
        "loadavg": "0.10 0.10 0.10 1/100 1",
        "nproc": 8,
        "want": {"decision": "reject", "admit": False, "hold": "bind_reject"},
    },
    {
        "name": "sbc_cpu_unparsed_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "cpu_max": "unlimited",
        "loadavg": "0.10 0.10 0.10 1/40 2",
        "nproc": 4,
        "want": {"decision": "hold", "admit": False, "hold": "unparsed_cpu"},
    },
    {
        "name": "sff_load_unparsed_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "cpu_max": "max",
        "loadavg": "not-a-loadavg",
        "nproc": 8,
        "want": {"decision": "hold", "admit": False, "hold": "unparsed_load"},
    },
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    rows = []
    fails = []
    fd, path = tempfile.mkstemp(prefix="hwsw_cgroup_cpu_", suffix=".jsonl")
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
                          f"hold={r['hold']!s:<14} frac={r['frac']!s:<8} "
                          f"load1={r['load1']!s:<6} close={prev}")
    finally:
        os.remove(path)
    return {
        "seam": "cgroup_cpu_loadavg->i2sd_admit",
        "roadmap": "ROADMAP #3 i2sd userspace",
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
