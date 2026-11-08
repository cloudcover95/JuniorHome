#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/hwmon fan + power + input voltage -> i2sd userspace admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace — 127.0.0.1:8767. No
  docker.sock.", at the seam where board sensors exposed through the hwmon
  class (fan tachometer and pwm mode, power1_average/power1_cap, in0
  supply rail, temp*_crit_alarm) decide whether i2sd may admit work and
  with how many threads. Hardware classes: SBC arm64 (<=12 W, JuniorOS T4)
  and SFF host (JuniorHome T0). Distinct from the thermal-zone/cpufreq
  seam (this file never reads /sys/class/thermal or cpufreq), the
  power_supply seam (no Mains/USB/Battery supply class), and the cgroup
  cpu.max seam (no cgroup files). Not T13 (overnight owns it). Not item 2
  (no GGUF), not item 4 (no Asahi/MLX), not item 5 (ue5_launch False).
  No other team file or earlier hwsw_* seam is imported, read, or wrapped.

What it emulates
  In-memory fixtures shaped like hwmon attribute files (all integers as the
  kernel writes them: fan RPM, pwmN_enable 0/1/2+, power in microwatts,
  voltage in millivolts, alarms 0/1), plus the hardware class, the i2sd
  bind string, and the wanted thread count. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767; any other (incl. 0.0.0.0) ->
     reject (bind_reject). docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff"; else hold (bad_class).
  3. Any temp*_crit_alarm == 1 -> hold (crit_alarm). Unparsed alarm ->
     hold (unparsed).
  4. in0: if in0_input and in0_min both present and input < min -> hold
     (undervolt). SBC with no in0 at all -> soft (no_rail_sense); SFF
     without in0 is fine.
  5. Fan: if a fan is declared with pwm_enable >= 1 (kernel or auto
     control) and fan_input == 0 -> hold (fan_stall). SFF with no fan
     declared -> soft (fanless_sff). fan_input below fan_min (>0) ->
     soft (fan_low). Unparsed fan_input -> hold (unparsed).
  6. Power: cap = power1_cap if present else class cap (SBC 12 W,
     SFF 65 W). power1_average > cap -> hold (over_cap);
     > 0.90 * cap -> soft (near_cap). Unparsed -> soft (no_power_sense);
     absent -> treated as unknown, no flag.
  7. Threads: class max (SBC 4, SFF 8) capped by wanted; soft halves
     (floor, min 1); hold/reject gives 0. Admit only when decision is
     admit.
  Each row packs decision and check states as 2-bit trit codes (0/1/2,
  four per byte, I2_S order: first code in the low bits); rows chain with
  close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no hwmon writes (pwm, fan_target,
  power1_cap stay untouched), no i2c/SMBus/PMBus access, no lm-sensors
  binary, no systemd calls, no sockets (binds are strings; nothing
  listens or connects), no device opened, no network fetch, no
  subprocess, no model pull, no GGUF download. Rows go to a tempfile
  removed after the run; nothing under ~/.juniorhome. Flags mint,
  socket, layer_one, fetch, ue5_launch all False. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_hwmon_fan_power_vin_i2sd_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_hwmon_fan_power_vin_i2sd_admit_seam.py demo   # one line per fixture row
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
CLASS_CAP_UW = {"sbc": 12_000_000, "sff": 65_000_000}
CLASS_THREADS = {"sbc": 4, "sff": 8}
NEAR_CAP = 0.90
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


def num(v):
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return "bad"


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    hw = fx.get("hwmon", {})
    checks = {}  # name -> (state, flag)
    worst = "admit"
    first_flag = None

    def mark(name, state, flag=None):
        nonlocal worst, first_flag
        checks[name] = state
        if RANK[state] > RANK[worst]:
            worst = state
            first_flag = flag
        elif state != "admit" and RANK[state] == RANK[worst] and first_flag is None:
            first_flag = flag

    # 1. bind / docker.sock
    if fx.get("bind") != BIND_OK:
        mark("bind", "reject", "bind_reject")
    elif fx.get("docker_sock"):
        mark("bind", "reject", "docker_sock")
    else:
        mark("bind", "admit")
    # 2. class
    if cls not in CLASS_THREADS:
        mark("cls", "hold", "bad_class")
        cls_ok = False
    else:
        mark("cls", "admit")
        cls_ok = True
    # 3. crit alarms
    alarm_state = "admit"
    alarm_flag = None
    for k, v in sorted(hw.items()):
        if k.startswith("temp") and k.endswith("_crit_alarm"):
            n = num(v)
            if n == "bad":
                alarm_state, alarm_flag = "hold", alarm_flag or "unparsed"
            elif n == 1:
                alarm_state, alarm_flag = "hold", "crit_alarm"
    mark("alarm", alarm_state, alarm_flag)
    # 4. input rail
    vin, vmin = num(hw.get("in0_input")), num(hw.get("in0_min"))
    if vin == "bad" or vmin == "bad":
        mark("rail", "hold", "unparsed")
    elif vin is not None and vmin is not None and vin < vmin:
        mark("rail", "hold", "undervolt")
    elif vin is None and cls == "sbc":
        mark("rail", "soft", "no_rail_sense")
    else:
        mark("rail", "admit")
    # 5. fan
    if "fan1_input" in hw:
        rpm = num(hw.get("fan1_input"))
        fmin = num(hw.get("fan1_min"))
        pwm = num(hw.get("pwm1_enable"))
        if rpm == "bad" or fmin == "bad" or pwm == "bad":
            mark("fan", "hold", "unparsed")
        elif rpm == 0 and pwm is not None and pwm >= 1:
            mark("fan", "hold", "fan_stall")
        elif fmin and rpm is not None and rpm < fmin:
            mark("fan", "soft", "fan_low")
        else:
            mark("fan", "admit")
    elif cls == "sff":
        mark("fan", "soft", "fanless_sff")
    else:
        mark("fan", "admit")
    # 6. power
    avg = num(hw.get("power1_average"))
    cap = num(hw.get("power1_cap"))
    if cap in (None, "bad") or not cap or cap <= 0:
        cap = CLASS_CAP_UW.get(cls, 0)
    if avg == "bad":
        mark("power", "soft", "no_power_sense")
    elif avg is None or not cap:
        mark("power", "admit")
    elif avg > cap:
        mark("power", "hold", "over_cap")
    elif avg > NEAR_CAP * cap:
        mark("power", "soft", "near_cap")
    else:
        mark("power", "admit")
    # 7. threads
    want = fx.get("want_threads", 0)
    base = min(CLASS_THREADS.get(cls, 0), max(0, int(want))) if cls_ok else 0
    if worst == "admit":
        threads = base
    elif worst == "soft":
        threads = max(1, base // 2) if base else 0
    else:
        threads = 0
    trits = [DECISION_CODE[worst]] + [DECISION_CODE[checks[k]] for k in
                                      ("bind", "cls", "alarm", "rail", "fan", "power")]
    return {
        "name": fx["name"],
        "cls": cls,
        "decision": worst,
        "admit": worst == "admit",
        "hold": first_flag,
        "threads": threads,
        "cap_uw": cap,
        "checks": checks,
        "trits": trits,
    }


FIXTURES = [
    {"name": "sbc_clean_admit", "cls": "sbc", "bind": BIND_OK, "want_threads": 4,
     "hwmon": {"in0_input": "5080", "in0_min": "4750", "fan1_input": "3200", "pwm1_enable": "2",
               "power1_average": "6500000", "temp1_crit_alarm": "0"},
     "want": {"decision": "admit", "admit": True, "threads": 4, "hold": None}},
    {"name": "sbc_undervolt_hold", "cls": "sbc", "bind": BIND_OK, "want_threads": 4,
     "hwmon": {"in0_input": "4630", "in0_min": "4750", "power1_average": "5000000"},
     "want": {"decision": "hold", "threads": 0, "hold": "undervolt"}},
    {"name": "sbc_no_rail_soft", "cls": "sbc", "bind": BIND_OK, "want_threads": 4,
     "hwmon": {"power1_average": "4000000"},
     "want": {"decision": "soft", "threads": 2, "hold": "no_rail_sense"}},
    {"name": "sff_fan_stall_hold", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "0", "pwm1_enable": "2", "power1_average": "30000000"},
     "want": {"decision": "hold", "threads": 0, "hold": "fan_stall"}},
    {"name": "sff_fan_off_manual_ok", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "0", "pwm1_enable": "0", "power1_average": "20000000"},
     "want": {"decision": "admit", "threads": 8}},
    {"name": "sff_fan_low_soft", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "450", "fan1_min": "600", "pwm1_enable": "2"},
     "want": {"decision": "soft", "threads": 4, "hold": "fan_low"}},
    {"name": "sff_fanless_soft", "cls": "sff", "bind": BIND_OK, "want_threads": 6,
     "hwmon": {"power1_average": "15000000"},
     "want": {"decision": "soft", "threads": 3, "hold": "fanless_sff"}},
    {"name": "sff_near_cap_soft", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "1800", "pwm1_enable": "2", "power1_average": "61000000"},
     "want": {"decision": "soft", "threads": 4, "hold": "near_cap"}},
    {"name": "sff_board_cap_over_hold", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "1800", "pwm1_enable": "2", "power1_average": "36000000",
               "power1_cap": "35000000"},
     "want": {"decision": "hold", "threads": 0, "hold": "over_cap", "cap_uw": 35000000}},
    {"name": "sbc_over_class_cap_hold", "cls": "sbc", "bind": BIND_OK, "want_threads": 4,
     "hwmon": {"in0_input": "5100", "in0_min": "4750", "power1_average": "12500000"},
     "want": {"decision": "hold", "hold": "over_cap", "cap_uw": 12000000}},
    {"name": "sff_crit_alarm_hold", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "2400", "pwm1_enable": "2", "temp2_crit_alarm": "1"},
     "want": {"decision": "hold", "hold": "crit_alarm"}},
    {"name": "sbc_unparsed_fan_hold", "cls": "sbc", "bind": BIND_OK, "want_threads": 4,
     "hwmon": {"in0_input": "5000", "in0_min": "4750", "fan1_input": "N/A", "pwm1_enable": "2"},
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sff_unparsed_power_soft", "cls": "sff", "bind": BIND_OK, "want_threads": 8,
     "hwmon": {"fan1_input": "2000", "pwm1_enable": "2", "power1_average": "?"},
     "want": {"decision": "soft", "threads": 4, "hold": "no_power_sense"}},
    {"name": "sbc_wildcard_bind_reject", "cls": "sbc", "bind": "0.0.0.0:8767", "want_threads": 4,
     "hwmon": {"in0_input": "5100", "in0_min": "4750"},
     "want": {"decision": "reject", "admit": False, "threads": 0, "hold": "bind_reject"}},
    {"name": "sff_docker_sock_reject", "cls": "sff", "bind": BIND_OK, "docker_sock": True,
     "want_threads": 8, "hwmon": {"fan1_input": "2000", "pwm1_enable": "2"},
     "want": {"decision": "reject", "threads": 0, "hold": "docker_sock"}},
    {"name": "bad_class_hold", "cls": "laptop", "bind": BIND_OK, "want_threads": 4,
     "hwmon": {"fan1_input": "2000", "pwm1_enable": "2"},
     "want": {"decision": "hold", "threads": 0, "hold": "bad_class"}},
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    rows = []
    fails = []
    fd, path = tempfile.mkstemp(prefix="hwsw_hwmon_", suffix=".jsonl")
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
                    print(f"{r['name']:<26} decision={r['decision']:<7} hold={r['hold']!s:<15} "
                          f"threads={r['threads']} close={prev}")
    finally:
        os.remove(path)
    # Self-check of the pack order: first code in the low bits.
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    return {
        "seam": "hwmon_fan_power_vin->i2sd_admit",
        "roadmap": "ROADMAP #3 i2sd userspace 127.0.0.1:8767 no docker.sock",
        "rate": RATE,
        "bind": BIND_OK,
        "flags": FLAGS,
        "rows": len(rows),
        "pack_order_ok": pack_ok,
        "fail": fails,
        "pass": not fails and pack_ok and not any(FLAGS.values()),
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
