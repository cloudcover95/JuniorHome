#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/rtc + /etc/adjtime -> i2sd receipt stamp admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767. No
  docker.sock." The seam decides, from the real-time-clock layer alone,
  whether the i2sd userspace daemon may stamp an I2_S work receipt with a
  wall-clock time, must fall back to a boot-relative stamp, or must hold.
  Battery-less SBC boards (JuniorOS T4) often boot with no RTC or with an
  RTC reset to 1970; SFF hosts (JuniorHome T0) are expected to keep one.
  Not ROADMAP T13 (overnight owns it). Not item 5 (ue5_launch stays False).

What it emulates
  In-memory fixtures shaped like /sys/class/rtc/rtc0/{name,since_epoch,
  hctosys,wakealarm}, the third line of /etc/adjtime ("UTC" or "LOCAL"),
  and a system clock value with its sync flag (as timedatectl would report
  NTPSynchronized). Rules, applied in order:
  1. Bind: the i2sd bind string must be exactly 127.0.0.1:8767; anything
     else (0.0.0.0, another port) rejects the row.
  2. No rtc0 node: SBC -> "boot_relative" (stamp is monotonic seconds since
     boot, tagged rtc_absent); SFF -> "hold" (an SFF host without an RTC is
     a fault, not a design choice).
  3. adjtime third line "LOCAL" -> "hold" (receipts are UTC only); missing
     or unparsed adjtime counts as UTC, which is the kernel default.
  4. since_epoch unparsed or before 2024-01-01T00:00:00Z -> RTC unset:
     SBC with a synced system clock -> "wall" stamped from the system clock
     and flagged rtc_reset; otherwise -> "boot_relative" (SBC) / "hold" (SFF).
  5. |rtc - system| <= 2 s -> "wall". Drift 2..300 s with a synced system
     clock -> "wall" from the system clock, flagged rtc_drift. Drift > 300 s,
     or any drift > 2 s with an unsynced system clock -> "hold".
  6. A non-zero wakealarm in the past is reported as a stale_alarm flag only;
     the seam never writes it.
  Each row packs its decision and three check states as real 2-bit trit
  codes (0/1/2, four per byte, I2_S order: first code in the low bits), and
  rows are chained with close rule sha256(prev || pack || text)[:16] at
  rate 0.05.

What it does not touch
  No real /sys or /etc reads, no hwclock, no timedatectl, no chrony/ntpd,
  no RTC ioctl, no clock writes, no sockets (binds are strings; nothing
  listens or connects), no device opened, no model pull, no GGUF, no
  network fetch, no subprocess. It does not import or re-run any team file
  or any earlier hwsw_* seam (power, thermal, CPU ISA, memory, storage,
  USB-C, and so on). Rows go to a tempfile removed after the run; nothing
  is written under ~/.juniorhome. Flags: mint, socket, layer_one, fetch,
  ue5_launch all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_rtc_clock_i2sd_receipt_stamp_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_rtc_clock_i2sd_receipt_stamp_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
BIND = "127.0.0.1:8767"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
FLOOR_EPOCH = 1704067200  # 2024-01-01T00:00:00Z
TIGHT_S = 2
LOOSE_S = 300
GENESIS = "0" * 16
DECISION_CODE = {"hold": 0, "boot_relative": 1, "wall": 2}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack_trits(codes):
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            if c not in (0, 1, 2):
                raise ValueError("trit code must be 0, 1 or 2")
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_trits(data, n):
    return [(data[i // 4] >> (2 * (i % 4))) & 3 for i in range(n)]


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def decide(fx):
    """Return (decision, flags list, trit codes) for one fixture dict."""
    cls = fx["cls"]
    flags = []
    bind_ok = fx.get("bind") == BIND
    rtc = fx.get("rtc")
    adj = str(fx.get("adjtime_line3", "UTC") or "UTC").strip().upper()
    adj_ok = adj != "LOCAL"
    sys_t = _int(fx.get("system"))
    synced = bool(fx.get("synced"))
    rtc_state = 1  # 0 bad, 1 absent/unknown, 2 good
    if not bind_ok:
        dec = "hold"
        flags.append("bind_reject")
    elif rtc is None:
        dec = "boot_relative" if cls == "sbc" else "hold"
        flags.append("rtc_absent")
    elif not adj_ok:
        dec = "hold"
        flags.append("rtc_local_time")
        rtc_state = 0
    else:
        since = _int(rtc.get("since_epoch"))
        alarm = _int(rtc.get("wakealarm"))
        if alarm and sys_t is not None and alarm < sys_t:
            flags.append("stale_alarm")
        if since is None or since < FLOOR_EPOCH:
            rtc_state = 0
            flags.append("rtc_reset")
            if cls == "sbc" and synced and sys_t is not None and sys_t >= FLOOR_EPOCH:
                dec = "wall"
            else:
                dec = "boot_relative" if cls == "sbc" else "hold"
        elif sys_t is None:
            dec = "hold"
            flags.append("system_unparsed")
        else:
            drift = abs(since - sys_t)
            if drift <= TIGHT_S:
                dec = "wall"
                rtc_state = 2
            elif drift <= LOOSE_S and synced:
                dec = "wall"
                rtc_state = 1
                flags.append("rtc_drift")
            else:
                dec = "hold"
                rtc_state = 0
                flags.append("rtc_drift_hold")
    codes = [DECISION_CODE[dec], 2 if bind_ok else 0, 2 if adj_ok else 0, rtc_state]
    return dec, flags, codes


NOW = 1793167200  # fixed fixture time (2026-10-28T06:00:00Z); no real clock read
FIXTURES = [
    ("sff_good", {"cls": "sff", "bind": BIND, "rtc": {"name": "rtc_cmos", "since_epoch": NOW + 1, "hctosys": "1", "wakealarm": ""}, "system": NOW, "synced": True}, "wall", []),
    ("sbc_no_rtc", {"cls": "sbc", "bind": BIND, "rtc": None, "system": 1000, "synced": False}, "boot_relative", ["rtc_absent"]),
    ("sff_no_rtc", {"cls": "sff", "bind": BIND, "rtc": None, "system": NOW, "synced": True}, "hold", ["rtc_absent"]),
    ("sbc_reset_synced", {"cls": "sbc", "bind": BIND, "rtc": {"name": "rtc-pcf8563", "since_epoch": "86400", "hctosys": "1"}, "system": NOW, "synced": True}, "wall", ["rtc_reset"]),
    ("sbc_reset_unsynced", {"cls": "sbc", "bind": BIND, "rtc": {"name": "rtc-pcf8563", "since_epoch": "86400"}, "system": 90000, "synced": False}, "boot_relative", ["rtc_reset"]),
    ("sff_reset", {"cls": "sff", "bind": BIND, "rtc": {"name": "rtc_cmos", "since_epoch": "garbage"}, "system": NOW, "synced": True}, "hold", ["rtc_reset"]),
    ("sff_local_adjtime", {"cls": "sff", "bind": BIND, "adjtime_line3": "LOCAL", "rtc": {"name": "rtc_cmos", "since_epoch": NOW}, "system": NOW, "synced": True}, "hold", ["rtc_local_time"]),
    ("sff_drift_synced", {"cls": "sff", "bind": BIND, "rtc": {"name": "rtc_cmos", "since_epoch": NOW - 90}, "system": NOW, "synced": True}, "wall", ["rtc_drift"]),
    ("sff_drift_unsynced", {"cls": "sff", "bind": BIND, "rtc": {"name": "rtc_cmos", "since_epoch": NOW - 90}, "system": NOW, "synced": False}, "hold", ["rtc_drift_hold"]),
    ("sbc_drift_huge", {"cls": "sbc", "bind": BIND, "rtc": {"name": "rtc-pcf8563", "since_epoch": NOW - 4000}, "system": NOW, "synced": True}, "hold", ["rtc_drift_hold"]),
    ("sff_stale_alarm", {"cls": "sff", "bind": BIND, "rtc": {"name": "rtc_cmos", "since_epoch": NOW, "wakealarm": NOW - 600}, "system": NOW, "synced": True}, "wall", ["stale_alarm"]),
    ("sbc_bad_bind", {"cls": "sbc", "bind": "0.0.0.0:8767", "rtc": {"name": "rtc-pcf8563", "since_epoch": NOW}, "system": NOW, "synced": True}, "hold", ["bind_reject"]),
    ("sff_wrong_port", {"cls": "sff", "bind": "127.0.0.1:8765", "rtc": {"name": "rtc_cmos", "since_epoch": NOW}, "system": NOW, "synced": True}, "hold", ["bind_reject"]),
]


def run(verbose=False):
    prev = GENESIS
    rows, failures = [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_rtc_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for name, fx, want_dec, want_flags in FIXTURES:
                dec, flags, codes = decide(fx)
                pack = pack_trits(codes)
                if unpack_trits(pack, len(codes)) != codes:
                    failures.append(name + ":pack_roundtrip")
                text = json.dumps({"row": name, "cls": fx["cls"], "decision": dec, "flags": flags, "rate": RATE}, sort_keys=True)
                cur = close(prev, pack, text)
                row = {"row": name, "decision": dec, "flags": flags, "pack": pack.hex(), "prev": prev, "close": cur}
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
                if dec != want_dec or flags != want_flags:
                    failures.append("%s: got %s %s want %s %s" % (name, dec, flags, want_dec, want_flags))
                if verbose:
                    print(json.dumps(row, sort_keys=True))
                prev = cur
        with open(path) as fh:
            if sum(1 for _ in fh) != len(FIXTURES):
                failures.append("tempfile_row_count")
    finally:
        os.remove(path)
    # Recompute chain independently to confirm the close rule is deterministic.
    p = GENESIS
    for r in rows:
        dec, flags, codes = decide(dict(FIXTURES[[f[0] for f in FIXTURES].index(r["row"])][1]))
        cls = FIXTURES[[f[0] for f in FIXTURES].index(r["row"])][1]["cls"]
        text = json.dumps({"row": r["row"], "cls": cls, "decision": dec, "flags": flags, "rate": RATE}, sort_keys=True)
        p2 = close(p, pack_trits(codes), text)
        if p2 != r["close"]:
            failures.append(r["row"] + ":chain")
        p = p2
    if any(FLAGS.values()):
        failures.append("flags_not_false")
    return {"seam": "rtc_clock->i2sd_receipt_stamp", "roadmap": "ROADMAP #3 i2sd userspace", "bind": BIND,
            "rate": RATE, "flags": FLAGS, "rows": len(rows), "tip_close": prev,
            "pass": not failures, "failures": failures}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    result = run(verbose=demo)
    print(json.dumps(result, sort_keys=True))
    sys.exit(0 if result["pass"] else 1)
