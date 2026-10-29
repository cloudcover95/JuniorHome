#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/watchdog -> bitnetd.service supervision.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md, JuniorOS overlay: bitnetd.service bound to
  127.0.0.1:8765. The seam decides, from the watchdog layer alone, how the
  overlay may supervise bitnetd.service: "arm" (hardware runtime watchdog
  plus a service WatchdogSec), "soft" (service WatchdogSec only, no
  hardware runtime watchdog), or "hold" (leave supervision unchanged and
  report). SBC boards (JuniorOS T4) often ship without a usable watchdog;
  SFF hosts (JuniorHome T0) are expected to expose one (iTCO, sp5100, ...).
  Not ROADMAP T13 (overnight owns it). Not item 5 (ue5_launch stays False).

What it emulates
  In-memory fixtures shaped like /sys/class/watchdog/watchdog0/{identity,
  timeout,min_timeout,max_timeout,nowayout,bootstatus} plus a proposed
  bitnetd.service WatchdogSec= value and bind string. Rules, in order:
  1. Bind: must be exactly 127.0.0.1:8765; anything else rejects the row.
  2. WatchdogSec unparsed or outside 5..120 s -> "hold".
  3. No watchdog0 node: SBC -> "soft" (flag hw_absent); SFF -> "hold"
     (an SFF host without a watchdog is a fault, not a design choice).
  4. Any watchdog field unparsed -> "hold".
  5. bootstatus has WDIOF_CARDRESET (0x20) and nowayout=1 -> "hold"
     (last boot was a watchdog reset and the timer cannot be stopped,
     so arming again risks a reboot loop); CARDRESET alone is a flag.
  6. Runtime target: SBC 60 s, SFF 30 s, clamped into [min_timeout,
     max_timeout]. If max_timeout < 10 s -> "soft" (flag hw_too_short).
  7. Service WatchdogSec must be <= runtime/2 so bitnetd is pinged well
     inside the hardware window; otherwise -> "soft" (flag svc_too_long).
  8. Otherwise -> "arm". The ping interval reported is WatchdogSec/2.
  Each row packs its decision and three check states as real 2-bit trit
  codes (0/1/2, four per byte, I2_S order: first code in the low bits), and
  rows are chained with close rule sha256(prev || pack || text)[:16] at
  rate 0.05.

What it does not touch
  No real /sys reads, no /dev/watchdog open (opening it would arm a real
  timer), no WDIOC ioctl, no systemctl, no unit file writes, no sockets
  (binds are strings; nothing listens or connects), no device opened, no
  model pull, no GGUF, no network fetch, no subprocess. It does not import
  or re-run any team file or any earlier hwsw_* seam (RTC, power, thermal,
  CPU ISA, memory, storage, DRM, USB-C, and so on). Rows go to a tempfile
  removed after the run; nothing is written under ~/.juniorhome. Flags:
  mint, socket, layer_one, fetch, ue5_launch all False. JuniorSolana and
  the profile repo are untouched.

How to run
  python3 scripts/hwsw_watchdog_bitnetd_service_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_watchdog_bitnetd_service_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
BIND = "127.0.0.1:8765"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
WDIOF_CARDRESET = 0x20
RUNTIME_TARGET = {"SBC": 60, "SFF": 30}
DECISION_CODE = {"hold": 0, "soft": 1, "arm": 2}
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
        s = str(v).strip()
        return int(s, 16) if s.lower().startswith("0x") else int(s)
    except (TypeError, ValueError):
        return None


def decide(fx):
    """Return (decision, checks[bind, hw, svc], flags, runtime, ping)."""
    flags = []
    if fx.get("bind") != BIND:
        return "reject", [CHECK_FAIL, CHECK_SKIP, CHECK_SKIP], ["bad_bind"], None, None
    cls = fx["cls"]
    wsec = _int(fx.get("watchdog_sec"))
    if wsec is None or not 5 <= wsec <= 120:
        return "hold", [CHECK_PASS, CHECK_SKIP, CHECK_FAIL], ["svc_unparsed_or_range"], None, None
    ping = wsec / 2
    wd = fx.get("wd")
    if wd is None:
        if cls == "SBC":
            return "soft", [CHECK_PASS, CHECK_SKIP, CHECK_PASS], ["hw_absent"], None, ping
        return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_PASS], ["hw_absent_sff"], None, None
    vals = {k: _int(wd.get(k)) for k in ("timeout", "min_timeout", "max_timeout", "nowayout", "bootstatus")}
    if any(v is None for v in vals.values()):
        return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_PASS], ["wd_unparsed"], None, None
    cardreset = bool(vals["bootstatus"] & WDIOF_CARDRESET)
    if cardreset:
        flags.append("last_boot_watchdog_reset")
        if vals["nowayout"] == 1:
            return "hold", [CHECK_PASS, CHECK_FAIL, CHECK_PASS], flags + ["nowayout_loop_risk"], None, None
    if vals["nowayout"] == 1:
        flags.append("nowayout")
    if vals["max_timeout"] < 10:
        return "soft", [CHECK_PASS, CHECK_FAIL, CHECK_PASS], flags + ["hw_too_short"], None, ping
    runtime = min(max(RUNTIME_TARGET[cls], vals["min_timeout"]), vals["max_timeout"])
    if wsec > runtime / 2:
        return "soft", [CHECK_PASS, CHECK_PASS, CHECK_FAIL], flags + ["svc_too_long"], runtime, ping
    return "arm", [CHECK_PASS, CHECK_PASS, CHECK_PASS], flags, runtime, ping


def _wd(timeout="30", mn="1", mx="300", nowayout="0", bootstatus="0", identity="iTCO_wdt"):
    return {"identity": identity, "timeout": timeout, "min_timeout": mn, "max_timeout": mx,
            "nowayout": nowayout, "bootstatus": bootstatus}


FIXTURES = [
    ("sff_itco_arm", {"cls": "SFF", "bind": BIND, "watchdog_sec": "10", "wd": _wd()}, "arm", 30),
    ("sbc_dw_wdt_arm", {"cls": "SBC", "bind": BIND, "watchdog_sec": "20", "wd": _wd(identity="dw_wdt", mx="171")}, "arm", 60),
    ("sbc_no_watchdog_soft", {"cls": "SBC", "bind": BIND, "watchdog_sec": "20", "wd": None}, "soft", None),
    ("sff_no_watchdog_hold", {"cls": "SFF", "bind": BIND, "watchdog_sec": "10", "wd": None}, "hold", None),
    ("sbc_max_short_soft", {"cls": "SBC", "bind": BIND, "watchdog_sec": "5", "wd": _wd(identity="bcm2835-wdt", mx="8")}, "soft", None),
    ("sbc_clamped_svc_long_soft", {"cls": "SBC", "bind": BIND, "watchdog_sec": "20", "wd": _wd(identity="bcm2835-wdt", mx="15")}, "soft", 15),
    ("sff_cardreset_nowayout_hold", {"cls": "SFF", "bind": BIND, "watchdog_sec": "10", "wd": _wd(nowayout="1", bootstatus="0x20")}, "hold", None),
    ("sff_cardreset_only_arm", {"cls": "SFF", "bind": BIND, "watchdog_sec": "10", "wd": _wd(bootstatus="32")}, "arm", 30),
    ("sff_wd_unparsed_hold", {"cls": "SFF", "bind": BIND, "watchdog_sec": "10", "wd": _wd(mx="")}, "hold", None),
    ("svc_out_of_range_hold", {"cls": "SFF", "bind": BIND, "watchdog_sec": "300", "wd": _wd()}, "hold", None),
    ("wildcard_bind_reject", {"cls": "SFF", "bind": "0.0.0.0:8765", "watchdog_sec": "10", "wd": _wd()}, "reject", None),
    ("wrong_port_reject", {"cls": "SBC", "bind": "127.0.0.1:8767", "watchdog_sec": "20", "wd": _wd()}, "reject", None),
]


def run_rows():
    rows, prev = [], "0" * 16
    for name, fx, _want, _rt in FIXTURES:
        dec, checks, flags, runtime, ping = decide(fx)
        codes = [DECISION_CODE.get(dec, 0)] + checks
        pack = pack_trits(codes)
        text = json.dumps({"row": name, "cls": fx["cls"], "decision": dec, "flags": flags,
                           "runtime_s": runtime, "ping_s": ping, "rate": RATE}, sort_keys=True)
        prev = close(prev, pack, text)
        rows.append({"row": name, "decision": dec, "codes": codes, "pack": pack.hex(),
                     "flags": flags, "runtime_s": runtime, "ping_s": ping, "close": prev})
    return rows, prev


def self_test():
    failures = []
    rows, tip = run_rows()
    for (name, _fx, want, want_rt), r in zip(FIXTURES, rows):
        if r["decision"] != want:
            failures.append(f"{name}: decision {r['decision']} != {want}")
        if want_rt is not None and r["runtime_s"] != want_rt:
            failures.append(f"{name}: runtime {r['runtime_s']} != {want_rt}")
        if unpack_trits(bytes.fromhex(r["pack"]), len(r["codes"])) != r["codes"]:
            failures.append(f"{name}: trit pack round-trip failed")
        if r["decision"] == "reject" and "bad_bind" not in r["flags"]:
            failures.append(f"{name}: reject without bad_bind")
    rows2, tip2 = run_rows()
    if tip2 != tip:
        failures.append("close chain not deterministic")
    if any(FLAGS.values()):
        failures.append("a guard flag is True")
    fd, path = tempfile.mkstemp(prefix="hwsw_watchdog_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        with open(path) as fh:
            if sum(1 for _ in fh) != len(rows):
                failures.append("tempfile row count mismatch")
    finally:
        os.remove(path)
    return {"seam": "watchdog->bitnetd.service", "bind": BIND, "rate": RATE, "flags": FLAGS,
            "rows": len(rows), "tip_close": tip, "pass": not failures, "failures": failures}


def main(argv):
    if len(argv) > 1 and argv[1] == "demo":
        rows, _ = run_rows()
        for r in rows:
            print(f"{r['row']:<30} {r['decision']:<6} runtime={r['runtime_s']} ping={r['ping_s']} "
                  f"pack={r['pack']} flags={','.join(r['flags']) or '-'} close={r['close']}")
        return 0
    out = self_test()
    print(json.dumps(out, sort_keys=True))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
