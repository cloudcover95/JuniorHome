"""Chief weekly receipt port map: ROADMAP chief weekly vs its receipts.

Advances JuniorHome docs/ROADMAP.md named item, Automations (current) ON:
"JuniorCloud chief weekly Sun/Mon 17:00 MDT (dd0b1c93) - only weekly.
Writes docs/WEEKLY_AUDIT.md".

Read-only. This file never writes docs/WEEKLY_AUDIT.md (LLC-owned) and does
not import, wrap or re-check any scripts/ or web3node/ chief weekly file or
any ports/daily_audit_* file. It reads only the receipt header lines
("# Weekly audit <date> <HH:MM> MDT") and host:port mentions, then checks:

  * ROADMAP ON names the chief weekly: dd0b1c93, Sun/Mon 17:00, the
    WEEKLY_AUDIT.md target, and "only weekly" (Home).
  * Friday weekly 7716afb1 sits under PAUSED, not ON (Flagstaff).
  * Every receipt header lands on Sun or Mon in the 17:xx hour (FieldCore).
  * Consecutive receipts are 6-8 days apart, no doubles (FieldCore).
  * Each receipt has a same-day 16:45 daily receipt in AUTOMATION_AUDIT.md,
    so the 17:00 weekly follows the 16:45 daily (OSai, advisory if missing).
  * The newest receipt is no older than 8 days at --asof (OSai, advisory).
  * Next due chief weekly slot from --asof (JuniorLLM, info).
  * Every bitnetd :8765 / i2sd :8767 host mention in the receipts is
    127.0.0.1, and 0.0.0.0 never binds those ports (JuniorLLM).
  * No .github directory; path flags held (Home).

Each check maps to one port (Home, OSai, FieldCore, Flagstaff, JuniorLLM)
and is signed into one chain, in-process, no sockets, no fetch.
Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false, fetch false, layer_one false, ue5_launch false,
bind 127.0.0.1. Close rule: sha256(prev || pack || text)[:16], trits as
2-bit codes (0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.

Usage: python3 ports/chief_weekly_receipt_port_map.py [--selftest]
       [--root DIR] [--asof YYYY-MM-DD]
Exit 0 when no check fails (advisories allowed), 1 otherwise.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
FETCH = False
LAYER_ONE = False
UE5_LAUNCH = False
MODEL_PULL = False
BIND = "127.0.0.1"
LOOPBACK_PORTS = {"bitnetd": 8765, "i2sd": 8767}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = "ROADMAP.md Automations ON: JuniorCloud chief weekly Sun/Mon 17:00 MDT (dd0b1c93)"
CHIEF_ID = "dd0b1c93"
FRIDAY_ID = "7716afb1"
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
TRIT_PACK = (1, -1, 0, 1, 1, 0, -1, 1)
_ENC = {0: 0, 1: 1, -1: 2}

_WEEKLY = re.compile(r"^# Weekly audit (\d{4}-\d{2}-\d{2}) ([0-2]\d):([0-5]\d) MDT", re.M)
_DAILY = re.compile(r"^# Automation audit (\d{4}-\d{2}-\d{2})[^\n]*16:4\d", re.M)
_HOSTPORT = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}):(876[57])\b")


def pack(trits) -> bytes:
    out, acc, n = bytearray(), 0, 0
    for t in trits:
        acc = (acc << 2) | _ENC[t]
        n += 1
        if n == 4:
            out.append(acc)
            acc, n = 0, 0
    if n:
        out.append(acc << (2 * (4 - n)))
    return bytes(out)


PACK = pack(TRIT_PACK)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def parse_roadmap(text: str) -> dict:
    on, paused, mode, in_block = [], [], None, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## "):
            in_block = s.lower().startswith("## automations")
            mode = None
            continue
        if not in_block:
            continue
        if s.startswith("ON:"):
            mode = "on"
        elif s.startswith("PAUSED:"):
            mode = "paused"
        elif s.startswith("-") and mode:
            (on if mode == "on" else paused).append(s)
    return {"on": on, "paused": paused}


def receipts(weekly: str) -> list:
    out = []
    for m in _WEEKLY.finditer(weekly):
        d = dt.date.fromisoformat(m.group(1))
        out.append((d, int(m.group(2)), int(m.group(3))))
    return out


def next_due(asof: dt.date) -> dt.date:
    d = asof
    while d.weekday() not in (6, 0):  # Sun=6, Mon=0
        d += dt.timedelta(days=1)
    return d


def evaluate(roadmap: str, weekly: str, daily: str, has_github: bool,
             asof: dt.date) -> dict:
    rm = parse_roadmap(roadmap)
    rc = receipts(weekly)
    daily_dates = {dt.date.fromisoformat(x) for x in _DAILY.findall(daily)}
    checks = []

    def add(port, name, status, note):
        checks.append({"port": port, "check": name, "status": status, "note": note})

    chief = [s for s in rm["on"] if CHIEF_ID in s]
    ok = bool(chief) and all(k in chief[0] for k in ("Sun/Mon", "17:00", "WEEKLY_AUDIT.md", "only weekly"))
    add("Home", "roadmap_on_names_chief_weekly", "pass" if ok else "fail",
        "ON line: %s" % (chief[0][:60] if chief else None))
    fri_on = any(FRIDAY_ID in s for s in rm["on"])
    fri_paused = any(FRIDAY_ID in s for s in rm["paused"])
    add("Flagstaff", "friday_weekly_paused", "pass" if fri_paused and not fri_on else "fail",
        "7716afb1 paused=%s on=%s" % (fri_paused, fri_on))
    add("FieldCore", "receipts_found", "pass" if rc else "fail", "%d receipt headers" % len(rc))
    bad_day = [str(d) for d, h, _ in rc if d.weekday() not in (6, 0) or h != 17]
    add("FieldCore", "receipts_sun_mon_17xx", "pass" if not bad_day else "fail",
        "off-slot=%s" % bad_day)
    dates = sorted(d for d, _, _ in rc)
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    bad_gap = [g for g in gaps if not 6 <= g <= 8]
    add("FieldCore", "receipt_cadence_weekly", "pass" if not bad_gap else "fail",
        "gaps=%s" % gaps)
    no_daily = [str(d) for d in dates if d not in daily_dates]
    add("OSai", "daily_1645_precedes_weekly", "advisory" if no_daily else "pass",
        "weekly dates without a 16:45 daily receipt: %s" % no_daily)
    newest = dates[-1] if dates else None
    age = (asof - newest).days if newest else None
    add("OSai", "newest_receipt_fresh",
        "pass" if age is not None and 0 <= age <= 8 else "advisory",
        "newest=%s age_days=%s asof=%s" % (newest, age, asof))
    nd = next_due(asof)
    add("JuniorLLM", "next_due_slot", "pass",
        "next Sun/Mon 17:00 MDT on or after asof: %s (%s)" % (nd, nd.strftime("%a")))
    hp = _HOSTPORT.findall(weekly)
    off = sorted({"%s:%s" % x for x in hp if x[0] != BIND})
    add("JuniorLLM", "receipt_loopback_ports", "pass" if not off else "fail",
        "%d host:port mentions, non-loopback=%s" % (len(hp), off))
    add("JuniorLLM", "loopback_port_map",
        "pass" if BIND == "127.0.0.1" and LOOPBACK_PORTS == {"bitnetd": 8765, "i2sd": 8767}
        else "fail", "bitnetd %s:8765, i2sd %s:8767" % (BIND, BIND))
    add("Home", "no_github_dir", "fail" if has_github else "pass", ".github absent")
    flags = (MINT, SOCKET, FETCH, LAYER_ONE, UE5_LAUNCH, MODEL_PULL)
    add("Home", "path_flags", "pass" if not any(flags) and RATE == 0.05 and OPT_IN
        and ANTICON and POOL_HELD else "fail", "rate 0.05, all gates false")

    prev, chain = GENESIS, []
    for c in checks:
        prev = close(prev, PACK, "%s|%s|%s" % (c["port"], c["check"], c["status"]))
        chain.append(prev)
    counts = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "advisory", "fail")}
    return {"item": ROADMAP_ITEM, "asof": str(asof), "receipts": [str(d) for d in dates],
            "checks": checks, "counts": counts, "close": chain[-1] if chain else GENESIS,
            "ports": list(PORTS), "bind": BIND}


_FIX_RM = ("# r\n## Automations (current)\nON:\n- 16:45 daily audit\n"
           "- **JuniorCloud chief weekly** Sun/Mon 17:00 MDT (`dd0b1c93`) \u2014 only weekly. "
           "Writes `docs/WEEKLY_AUDIT.md`.\n\nPAUSED:\n- 07:00, 13:00\n"
           "- Friday weekly `7716afb1` (merged into chief)\n\n## Order\n1. x\n")
_FIX_WK = ("# Weekly audit 2026-01-05 17:06 MDT\nbitnetd 127.0.0.1:8765\n\n"
           "# Weekly audit 2026-01-12 17:04 MDT\ni2sd 127.0.0.1:8767; never 0.0.0.0 bind\n")
_FIX_DA = "# Automation audit 2026-01-05 16:45 MDT\n# Automation audit 2026-01-12 16:45 MDT (tip a)\n"


def selftest() -> int:
    assert pack((1, 0, -1, 1)) == bytes([0b01001001])
    assert pack((1,)) == bytes([0b01000000])
    assert close(GENESIS, PACK, "x") == hashlib.sha256(GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    r = evaluate(_FIX_RM, _FIX_WK, _FIX_DA, False, dt.date(2026, 1, 15))
    assert r["counts"]["fail"] == 0 and r["counts"]["advisory"] == 0, r["checks"]
    assert next_due(dt.date(2026, 1, 15)) == dt.date(2026, 1, 18)  # Thu -> Sun
    bad = _FIX_WK + "# Weekly audit 2026-01-16 17:00 MDT\nbitnetd 0.0.0.0:8765\n"
    r2 = evaluate(_FIX_RM, bad, _FIX_DA, True, dt.date(2026, 1, 16))
    names = {c["check"] for c in r2["checks"] if c["status"] == "fail"}
    assert {"receipts_sun_mon_17xx", "receipt_cadence_weekly", "receipt_loopback_ports",
            "no_github_dir"} <= names, names
    r3 = evaluate(_FIX_RM.replace("PAUSED:\n- 07:00, 13:00\n- Friday", "- Friday"),
                  _FIX_WK, "", False, dt.date(2026, 2, 1))
    st = {c["check"]: c["status"] for c in r3["checks"]}
    assert st["friday_weekly_paused"] == "fail"
    assert st["daily_1645_precedes_weekly"] == "advisory"
    assert st["newest_receipt_fresh"] == "advisory"
    assert r["close"] != r2["close"]
    print("selftest ok close=%s" % r["close"])
    return 0


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def main(argv) -> int:
    if "--selftest" in argv:
        return selftest()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    asof = dt.date.today()
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    if "--asof" in argv:
        asof = dt.date.fromisoformat(argv[argv.index("--asof") + 1])
    r = evaluate(_read(os.path.join(root, "docs", "ROADMAP.md")),
                 _read(os.path.join(root, "docs", "WEEKLY_AUDIT.md")),
                 _read(os.path.join(root, "docs", "AUTOMATION_AUDIT.md")),
                 os.path.isdir(os.path.join(root, ".github")), asof)
    print(json.dumps(r, indent=1))
    return 1 if r["counts"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
