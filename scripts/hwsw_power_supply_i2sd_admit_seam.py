#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/power_supply -> i2sd admit and watt budget.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767. No
  docker.sock." The seam decides, from the power layer alone, whether the
  i2sd userspace daemon may admit I2_S work on a box and with what watt
  budget. Hardware classes: SBC (arm64, <=12 W, JuniorOS T4) and SFF host
  (JuniorHome T0). Not ROADMAP T13 (overnight owns it). Not item 5: the
  watts figure here is the i2sd power budget only and never feeds a UE5
  launch decision (ue5_launch stays False).

What it emulates
  In-memory fixtures shaped like /sys/class/power_supply/<name>/ attribute
  files: type (Mains, USB, Battery), online, status, capacity,
  voltage_now / current_max (uV / uA), power_now (uW), usb_type.
  1. Sources: a Mains or USB supply with online=1 contributes
     voltage_now * current_max as its offered watts (or power_now when the
     pair is missing). usb_type "[PD]" may offer more than 5 V; plain
     "[SDP]" is capped at 5 V x 0.5 A, "[DCP]"/"[CDP]" at 5 V x 1.5 A.
     Unparsed numbers count as 0 W.
  2. Battery: discharging with capacity < 20 -> hold; capacity < 50 on
     battery only -> eco; status "Unknown" or unparsed capacity with no
     online source -> hold. A healthy battery-only box is offered its class
     cap (no floor check, since no external supply is drawn).
  3. Class floor: SBC needs >= 7.5 W offered, SFF needs >= 25 W offered when
     on external power; below the floor -> hold (brown-out guard).
  4. Budget: SBC min(offered, 12) W, SFF min(offered, 65) W, eco halves it,
     and threads = SBC 1 per 3 W capped 4, SFF 1 per 8 W capped 8.
  5. Bind check: i2sd bind must be exactly the string 127.0.0.1:8767; a
     0.0.0.0 or other bind rejects the row.
  6. Correctness guard: a fixed ternary block dot product split over the
     admitted thread count must equal the scalar reference.
  One row per fixture is chained with close rule sha256(prev || pack || text)[:16]
  at rate 0.05, where pack is real 2-bit trit codes (0/1/2, four per byte,
  I2_S order) built from source, battery, floor, and bind states.

What it does not touch
  No real /sys reads, no upower, no D-Bus, no PD controller, no charger
  writes, no sockets (binds are strings; nothing listens or connects), no
  device opened, no model pull, no GGUF, no network fetch, no subprocess.
  It does not import or re-run any team file or any earlier hwsw_* seam
  (thermal/cpufreq, CPU ISA, memory, storage, USB-C, and so on).
  Rows go to a tempfile removed after the run; nothing under ~/.juniorhome.
  Flags: mint, socket, layer_one, fetch, ue5_launch all False.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_power_supply_i2sd_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_power_supply_i2sd_admit_seam.py demo   # one line per fixture row
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
FLOOR_W = {"sbc": 7.5, "sff": 25.0}
CAP_W = {"sbc": 12.0, "sff": 65.0}
W_PER_THREAD = {"sbc": 3.0, "sff": 8.0}
THREAD_CAP = {"sbc": 4, "sff": 8}
USB_CAPS_W = {"SDP": 2.5, "DCP": 7.5, "CDP": 7.5}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}

W_CODES = [(i * 5 + 1) % 3 for i in range(96)]
ACTS = [((i * 29) % 255) - 127 for i in range(96)]


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


def dot_scalar(codes: list, acts: list) -> int:
    return sum((c - 1) * a for c, a in zip(codes, acts))


def dot_split(codes: list, acts: list, threads: int) -> int:
    t = max(1, threads)
    step = (len(codes) + t - 1) // t
    total = 0
    for s in range(0, len(codes), step):
        total += sum((c - 1) * a for c, a in zip(codes[s:s + step], acts[s:s + step]))
    return total


def num(v):
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def usb_kind(raw: str) -> str:
    for tok in str(raw).split():
        if tok.startswith("[") and tok.endswith("]"):
            return tok[1:-1]
    return ""


def source_watts(ps: dict) -> float:
    if str(ps.get("type", "")).strip() not in ("Mains", "USB"):
        return 0.0
    if num(ps.get("online", "0")) != 1:
        return 0.0
    uv, ua = num(ps.get("voltage_now", "")), num(ps.get("current_max", ""))
    if uv is not None and ua is not None and uv >= 0 and ua >= 0:
        w = uv * ua / 1e12
    else:
        pw = num(ps.get("power_now", ""))
        w = pw / 1e6 if pw is not None and pw >= 0 else 0.0
    if str(ps.get("type")).strip() == "USB":
        kind = usb_kind(ps.get("usb_type", ""))
        if kind in USB_CAPS_W:
            w = min(w, USB_CAPS_W[kind])
        elif kind != "PD":
            w = min(w, USB_CAPS_W["SDP"])
    return round(w, 3)


def evaluate(fx: dict) -> dict:
    cls = fx["cls"]
    supplies = fx["supplies"]
    offered = round(sum(source_watts(p) for p in supplies), 3)
    external = offered > 0
    batt = [p for p in supplies if str(p.get("type", "")).strip() == "Battery"]
    batt_state = "none"
    eco = False
    hold_reason = ""
    if batt:
        b = batt[0]
        cap = num(b.get("capacity", ""))
        status = str(b.get("status", "")).strip()
        if not external:
            if cap is None or status == "Unknown":
                batt_state, hold_reason = "unknown", "battery_unparsed"
            elif status == "Discharging" and cap < 20:
                batt_state, hold_reason = "low", "battery_low"
            elif cap < 50:
                batt_state, eco = "eco", True
            else:
                batt_state = "ok"
            if not hold_reason:
                # battery-only: treat the class cap as offered from the cell
                offered = CAP_W[cls]
        else:
            batt_state = "charging" if status in ("Charging", "Full", "Not charging") else "ok"
    elif not external:
        hold_reason = "no_power"
    floor_ok = True
    if external and offered < FLOOR_W[cls]:
        floor_ok, hold_reason = False, hold_reason or "below_floor"
    bind_ok = fx.get("bind") == BIND_OK
    if not bind_ok:
        hold_reason = hold_reason or "bind_reject"
    admit = not hold_reason
    budget = 0.0
    threads = 0
    if admit:
        budget = min(offered, CAP_W[cls])
        if eco:
            budget /= 2
        threads = max(1, min(THREAD_CAP[cls], int(budget // W_PER_THREAD[cls])))
        budget = round(budget, 3)
    dot_ok = dot_split(W_CODES, ACTS, threads) == dot_scalar(W_CODES, ACTS)
    trits = [
        2 if external else (1 if batt else 0),
        {"none": 1, "ok": 2, "charging": 2, "eco": 1, "low": 0, "unknown": 0}[batt_state],
        2 if floor_ok else 0,
        2 if bind_ok else 0,
        2 if admit else 0,
        min(2, threads // 3),
    ]
    return {"name": fx["name"], "cls": cls, "offered_w": offered, "battery": batt_state,
            "floor_ok": floor_ok, "bind_ok": bind_ok, "admit": admit,
            "hold": hold_reason or None, "budget_w": budget, "threads": threads,
            "dot_ok": dot_ok, "trits": trits}


FIXTURES = [
    {"name": "sbc_pd_15w", "cls": "sbc", "bind": BIND_OK, "supplies": [
        {"type": "USB", "online": "1", "usb_type": "C [PD] PD_PPS", "voltage_now": "9000000", "current_max": "1670000"}],
     "want": {"admit": True, "budget_w": 12.0, "threads": 4}},
    {"name": "sbc_sdp_laptop_port", "cls": "sbc", "bind": BIND_OK, "supplies": [
        {"type": "USB", "online": "1", "usb_type": "[SDP] DCP CDP", "voltage_now": "5000000", "current_max": "3000000"}],
     "want": {"admit": False, "hold": "below_floor"}},
    {"name": "sbc_dcp_brick", "cls": "sbc", "bind": BIND_OK, "supplies": [
        {"type": "USB", "online": "1", "usb_type": "SDP [DCP]", "voltage_now": "5000000", "current_max": "2000000"}],
     "want": {"admit": True, "budget_w": 7.5, "threads": 2}},
    {"name": "sff_mains_90w", "cls": "sff", "bind": BIND_OK, "supplies": [
        {"type": "Mains", "online": "1", "power_now": "90000000"}],
     "want": {"admit": True, "budget_w": 65.0, "threads": 8}},
    {"name": "sff_battery_low", "cls": "sff", "bind": BIND_OK, "supplies": [
        {"type": "Mains", "online": "0"},
        {"type": "Battery", "status": "Discharging", "capacity": "12"}],
     "want": {"admit": False, "hold": "battery_low"}},
    {"name": "sff_battery_eco", "cls": "sff", "bind": BIND_OK, "supplies": [
        {"type": "Mains", "online": "0"},
        {"type": "Battery", "status": "Discharging", "capacity": "35"}],
     "want": {"admit": True, "budget_w": 32.5, "threads": 4}},
    {"name": "sbc_battery_unparsed", "cls": "sbc", "bind": BIND_OK, "supplies": [
        {"type": "Battery", "status": "Unknown", "capacity": "n/a"}],
     "want": {"admit": False, "hold": "battery_unparsed"}},
    {"name": "sff_wildcard_bind", "cls": "sff", "bind": "0.0.0.0:8767", "supplies": [
        {"type": "Mains", "online": "1", "power_now": "120000000"}],
     "want": {"admit": False, "hold": "bind_reject"}},
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    rows = []
    fails = []
    fd, path = tempfile.mkstemp(prefix="hwsw_power_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for fx in FIXTURES:
                r = evaluate(fx)
                text = json.dumps({k: v for k, v in r.items() if k != "trits"}, sort_keys=True)
                prev = close(prev, pack2(r["trits"]), text)
                r["close"] = prev
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                ok = r["dot_ok"] and all(r.get(k) == v for k, v in fx["want"].items())
                if not ok:
                    fails.append(fx["name"])
                rows.append(r)
                if verbose:
                    print(f"{r['name']:<22} admit={r['admit']!s:<5} hold={r['hold']!s:<16} "
                          f"offered={r['offered_w']:>7} budget={r['budget_w']:>6} threads={r['threads']} close={prev}")
    finally:
        os.remove(path)
    return {"seam": "power_supply->i2sd", "roadmap": "ROADMAP #3 i2sd userspace", "rate": RATE,
            "bind": BIND_OK, "flags": FLAGS, "rows": len(rows), "fail": fails,
            "pass": not fails, "tip_close": prev}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    if not demo:
        print(json.dumps(out, indent=2))
    sys.exit(0 if out["pass"] else 1)
