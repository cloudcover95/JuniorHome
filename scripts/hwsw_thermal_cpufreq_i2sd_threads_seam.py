#!/usr/bin/env python3
"""Hardware-to-software seam emulator: thermal zones + cpufreq -> i2sd userspace thread budget.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace — 127.0.0.1:8767. No
  docker.sock.", on the hardware class rows SBC (arm64, <=12 W -> JuniorOS T4)
  and SFF host (x86_64 or arm64 -> JuniorHome T0). The seam answers one
  question: when the board gets hot or the governor clocks it down, how many
  I2_S worker threads may i2sd run, and when must it hold new work? Rule:
  heat and clock-down only ever lower the budget; i2sd never kills a running
  request and never raises threads past the class cap. Whatever the thread
  count, the ternary dot product must equal the single-thread reference
  exactly. Not ROADMAP T13 (overnight owns it). Not a re-derivation of any
  earlier seam file: no other team file is imported or read.

What it emulates
  In-memory fixtures shaped like sysfs text:
    /sys/class/thermal/thermal_zoneN/{type,temp,trip_point_0_temp,
      trip_point_0_type,trip_point_1_temp,trip_point_1_type}  (millidegree C)
    /sys/devices/system/cpu/cpufreq/policy0/{scaling_cur_freq,
      cpuinfo_max_freq,scaling_governor}  (kHz)
  plus a watt class, a core count, the previous state and the i2sd bind.
  1. Parse: temps are integers in m°C; the hottest zone wins. A zone whose
     temp does not parse is treated as at its critical trip (fail safe).
  2. Thermal state: temp >= critical trip -> "hold" (0 new threads, running
     work finishes); temp >= passive trip -> "throttle" (half the cap, at
     least 1); otherwise "cool". Hysteresis: leaving throttle/hold needs the
     temp to fall 5000 m°C below the trip that was crossed.
  3. Clock-down: cur/max < 0.60 removes one more thread (floor 1 unless
     holding). Governor "powersave" on an SBC caps at 2.
  4. Caps: SBC 4, SFF 8, never above cores.
  5. Bind: i2sd must be exactly 127.0.0.1:8767; anything else rejects the row.
  6. Exactness: a fixed ternary weight block (codes 0/1/2 = -1/0/+1, four per
     byte, low bits first) is dotted with int8 activations split across the
     chosen thread count (stdlib, sequential slices standing in for threads);
     the row passes only if the sum equals the 1-thread scalar reference.
  One row per fixture is chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05, where pack is the real
  2-bit trit weight block plus trit codes for thermal state and bind.

What it does not touch
  No real /sys reads, no thermal or cpufreq writes, no governor change, no
  sockets (nothing binds or connects), no device opened, no model pull, no
  GGUF, no network fetch, no subprocess. mint/socket/layer_one/fetch/
  ue5_launch are all false. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_thermal_cpufreq_i2sd_threads_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_thermal_cpufreq_i2sd_threads_seam.py demo   # one line per fixture row
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
HYST_MC = 5000
CLOCK_FLOOR = 0.60
CAPS = {"sbc": 4, "sff": 8}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}

# fixed ternary block (codes 0/1/2 = -1/0/+1) and int8 activations
W_CODES = [(i * 7 + 3) % 3 for i in range(96)]
ACTS = [((i * 37) % 255) - 127 for i in range(96)]


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes: list) -> bytes:
    codes = list(codes)
    while len(codes) % 4:
        codes.append(1)  # pad with the zero weight
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            assert c in (0, 1, 2)
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def dot_scalar(codes: list, acts: list) -> int:
    return sum((c - 1) * a for c, a in zip(codes, acts))


def dot_split(codes: list, acts: list, threads: int) -> int:
    n = len(codes)
    t = max(1, threads)
    step = (n + t - 1) // t
    total = 0
    for s in range(0, n, step):
        part = 0
        for c, a in zip(codes[s:s + step], acts[s:s + step]):
            part += (c - 1) * a
        total += part
    return total


def parse_int(text: str):
    try:
        return int(str(text).strip())
    except ValueError:
        return None


def zone_view(zone: dict) -> dict:
    trips = {}
    for k in (0, 1):
        ttype = str(zone.get(f"trip_point_{k}_type", "")).strip()
        tval = parse_int(zone.get(f"trip_point_{k}_temp", ""))
        if ttype and tval is not None:
            trips[ttype] = tval
    passive = trips.get("passive", 85000)
    critical = trips.get("critical", 105000)
    temp = parse_int(zone.get("temp", ""))
    bad = temp is None
    if bad:
        temp = critical
    return {"type": str(zone.get("type", "?")).strip(), "temp": temp,
            "passive": passive, "critical": critical, "unparsed": bad}


def thermal_state(z: dict, prev_state: str) -> str:
    t, p, c = z["temp"], z["passive"], z["critical"]
    if t >= c:
        return "hold"
    if prev_state == "hold" and t > c - HYST_MC:
        return "hold"
    if t >= p:
        return "throttle"
    if prev_state in ("throttle", "hold") and t > p - HYST_MC:
        return "throttle"
    return "cool"


def budget(fx: dict) -> dict:
    why = []
    if fx["bind"] != BIND_OK:
        return {"state": "reject", "threads": 0, "why": ["bind_not_127.0.0.1:8767"], "zone": None}
    zones = [zone_view(z) for z in fx["zones"]]
    hot = max(zones, key=lambda z: z["temp"] - z["passive"])
    if any(z["unparsed"] for z in zones):
        why.append("zone_unparsed_failsafe_critical")
    state = thermal_state(hot, fx.get("prev_state", "cool"))
    cap = min(CAPS[fx["cls"]], max(1, fx["cores"]))
    if state == "hold":
        return {"state": "hold", "threads": 0, "why": why + ["critical_trip"], "zone": hot["type"]}
    threads = cap if state == "cool" else max(1, cap // 2)
    if state == "throttle":
        why.append("passive_trip")
    cf = fx["cpufreq"]
    cur, mx = parse_int(cf["scaling_cur_freq"]), parse_int(cf["cpuinfo_max_freq"])
    if not cur or not mx or cur / mx < CLOCK_FLOOR:
        threads = max(1, threads - 1)
        why.append("clock_down")
    if fx["cls"] == "sbc" and str(cf.get("scaling_governor", "")).strip() == "powersave":
        if threads > 2:
            threads = 2
        why.append("sbc_powersave_cap2")
    threads = min(threads, cap)
    return {"state": state, "threads": threads, "why": why or ["nominal"], "zone": hot["type"]}


def z(t, temp, p=85000, c=105000):
    return {"type": t, "temp": str(temp), "trip_point_0_temp": str(p), "trip_point_0_type": "passive\n",
            "trip_point_1_temp": str(c), "trip_point_1_type": "critical\n"}


def cpuf(cur, mx, gov="schedutil"):
    return {"scaling_cur_freq": str(cur), "cpuinfo_max_freq": str(mx), "scaling_governor": gov + "\n"}


FIXTURES = [
    {"host": "sbc_cool", "cls": "sbc", "cores": 4, "bind": BIND_OK,
     "zones": [z("cpu-thermal", 52000)], "cpufreq": cpuf(1800000, 1800000),
     "expect": ("cool", 4)},
    {"host": "sbc_passive", "cls": "sbc", "cores": 4, "bind": BIND_OK,
     "zones": [z("cpu-thermal", 86500), z("gpu-thermal", 70000)], "cpufreq": cpuf(1500000, 1800000),
     "expect": ("throttle", 2)},
    {"host": "sbc_hysteresis_stays_throttled", "cls": "sbc", "cores": 4, "bind": BIND_OK, "prev_state": "throttle",
     "zones": [z("cpu-thermal", 82000)], "cpufreq": cpuf(1800000, 1800000),
     "expect": ("throttle", 2)},
    {"host": "sbc_critical_hold", "cls": "sbc", "cores": 4, "bind": BIND_OK,
     "zones": [z("cpu-thermal", 105500)], "cpufreq": cpuf(600000, 1800000),
     "expect": ("hold", 0)},
    {"host": "sbc_powersave_clockdown", "cls": "sbc", "cores": 4, "bind": BIND_OK,
     "zones": [z("cpu-thermal", 60000)], "cpufreq": cpuf(800000, 1800000, "powersave"),
     "expect": ("cool", 2)},
    {"host": "sff_cool_8", "cls": "sff", "cores": 12, "bind": BIND_OK,
     "zones": [z("x86_pkg_temp", 61000, 95000, 100000), z("acpitz", 45000)], "cpufreq": cpuf(3900000, 4200000),
     "expect": ("cool", 8)},
    {"host": "sff_passive_clockdown", "cls": "sff", "cores": 6, "bind": BIND_OK,
     "zones": [z("x86_pkg_temp", 96000, 95000, 100000)], "cpufreq": cpuf(2000000, 4200000),
     "expect": ("throttle", 2)},
    {"host": "sff_unparsed_zone_failsafe", "cls": "sff", "cores": 8, "bind": BIND_OK,
     "zones": [z("acpitz", "N/A"), z("x86_pkg_temp", 50000)], "cpufreq": cpuf(4200000, 4200000),
     "expect": ("hold", 0)},
    {"host": "sff_bad_bind", "cls": "sff", "cores": 8, "bind": "0.0.0.0:8767",
     "zones": [z("x86_pkg_temp", 50000)], "cpufreq": cpuf(4200000, 4200000),
     "expect": ("reject", 0)},
]

STATE_TRIT = {"cool": 2, "throttle": 1, "hold": 0, "reject": 0}


def run_rows() -> list:
    prev = GENESIS
    rows = []
    wpack = pack2(W_CODES)
    ref = dot_scalar(W_CODES, ACTS)
    for fx in FIXTURES:
        b = budget(fx)
        dot = dot_split(W_CODES, ACTS, b["threads"]) if b["threads"] else None
        exact = (dot == ref) if dot is not None else True
        body = {"host": fx["host"], "cls": fx["cls"], "state": b["state"], "threads": b["threads"],
                "why": b["why"], "zone": b["zone"], "dot": dot, "exact": exact, "rate": RATE}
        text = json.dumps(body, sort_keys=True)
        pack = wpack + pack2([STATE_TRIT[b["state"]], 2 if fx["bind"] == BIND_OK else 0])
        c = close(prev, pack, text)
        rows.append({"prev": prev, "body": body, "text": text, "pack": pack.hex(), "close": c,
                     "expect": fx["expect"]})
        prev = c
    return rows


def self_test() -> dict:
    rows = run_rows()
    results = []
    for r in rows:
        b = r["body"]
        want_state, want_threads = r["expect"]
        ok = (b["state"] == want_state and b["threads"] == want_threads and b["exact"]
              and b["threads"] <= CAPS[b["cls"]])
        results.append({"host": b["host"], "state": b["state"], "threads": b["threads"], "ok": ok})
    chain_ok = True
    prev = GENESIS
    for r in rows:
        chain_ok &= (r["prev"] == prev and close(prev, bytes.fromhex(r["pack"]), r["text"]) == r["close"])
        prev = r["close"]
    # monotonic: heating the same SBC fixture never raises threads
    mono_ok = True
    last = None
    for t in range(40000, 110001, 2500):
        fx = dict(FIXTURES[0], zones=[z("cpu-thermal", t)])
        n = budget(fx)["threads"]
        if last is not None and n > last:
            mono_ok = False
        last = n
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for r in rows:
                fh.write(json.dumps({"close": r["close"], "body": r["body"]}, sort_keys=True) + "\n")
        with open(path) as fh:
            disk_ok = [json.loads(line)["close"] for line in fh] == [r["close"] for r in rows]
    finally:
        os.remove(path)
    flags_ok = all(v is False for v in FLAGS.values())
    passed = all(x["ok"] for x in results) and chain_ok and mono_ok and disk_ok and flags_ok
    return {"seam": "thermal_zone+cpufreq->i2sd_thread_budget", "rate": RATE,
            "dot_ref": dot_scalar(W_CODES, ACTS), "rows": results, "chain_ok": chain_ok,
            "monotonic_ok": mono_ok, "disk_ok": disk_ok, "flags": FLAGS,
            "tip_close": rows[-1]["close"], "pass": passed}


def main(argv: list) -> int:
    if len(argv) > 1 and argv[1] == "demo":
        for r in run_rows():
            b = r["body"]
            print(json.dumps({"host": b["host"], "state": b["state"], "threads": b["threads"],
                              "why": b["why"], "close": r["close"]}, sort_keys=True))
        return 0
    res = self_test()
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
