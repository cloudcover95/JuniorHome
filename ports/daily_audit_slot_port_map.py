"""16:45 daily audit slot port map: ROADMAP Automations vs AUTOMATION_AUDIT.md.

Advances JuniorHome docs/ROADMAP.md named item, Automations (current) ON:
"16:45 daily audit" (writes docs/AUTOMATION_AUDIT.md).

New angle (not the declarative facts in ports/daily_audit_handshake.py,
which this file does not import, wrap or re-check): a read-only port map
that parses the two docs the audit slot lives in and confirms they agree.

  * ROADMAP lists "16:45 daily audit" under ON.
  * The newest "# Automation audit <date> 16:45 MDT" entry in
    docs/AUTOMATION_AUDIT.md carries a daily 16:45 row.
  * Every ON slot in ROADMAP (01:00, 19:00, 16:45, 17:00 weekly) has a row
    in that newest entry.
  * PAUSED slots (07:00, 13:00) have no row and the entry says
    "07:00 and 13:00 not scheduled" ("Do not recreate paused slots").
  * No .github directory (audit row: no CI yaml).
Slots the audit lists that ROADMAP does not (e.g. Sat weekly reports) and
weekday wording drift ("Sun/Mon" vs "Mon") are advisories, never fails.

Each check is mapped to one port (Home, OSai, FieldCore, Flagstaff,
JuniorLLM) and signed into one chain, in-process, no sockets, no fetch.
Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false, fetch false, layer_one false, ue5_launch false,
bind 127.0.0.1 (bitnetd 8765, i2sd 8767 listed in the port map only).
Close rule: sha256(prev || pack || text)[:16], trits as 2-bit codes
(0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.

Usage: python3 ports/daily_audit_slot_port_map.py [--selftest] [--root DIR]
Exit 0 when no check fails (advisories allowed), 1 otherwise.
"""
from __future__ import annotations

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
ROADMAP_ITEM = "ROADMAP.md Automations ON: 16:45 daily audit"
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
TRIT_PACK = (1, -1, 0, 1, 1, 0, -1, 1)
_ENC = {0: 0, 1: 1, -1: 2}

_TIME = re.compile(r"\b([0-2]\d:[0-5]\d)\b")
_ENTRY = re.compile(r"^# Automation audit (\d{4}-\d{2}-\d{2}) 16:45 MDT.*$", re.M)


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
    """Return ON / PAUSED slot times from the Automations (current) block."""
    on, paused, mode = [], [], None
    in_block = False
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
            continue
        if s.startswith("PAUSED:"):
            mode = "paused"
            continue
        if s.startswith("-") and mode:
            times = _TIME.findall(s)
            (on if mode == "on" else paused).extend(times)
            if mode == "on" and "daily audit" in s.lower():
                on.append("audit:" + (times[0] if times else "?"))
    return {"on": on, "paused": paused, "raw_weekly": "Sun/Mon" in text}


def parse_latest_audit(text: str) -> dict:
    heads = list(_ENTRY.finditer(text))
    if not heads:
        return {"date": None, "rows": [], "body": ""}
    h = heads[-1]
    body = text[h.end():]
    rows = []
    for line in body.splitlines():
        s = line.strip()
        if not s.startswith("|") or set(s) <= set("|-: "):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and cells[0].lower() in ("name", "check", "slot"):
            continue
        rows.append(cells)
    return {"date": h.group(1), "rows": rows, "body": body}


def _row_times(rows) -> list:
    out = []
    for r in rows:
        for c in r[1:3]:
            out.extend(_TIME.findall(c))
    return out


def evaluate(roadmap: str, audit: str, has_github: bool) -> dict:
    rm = parse_roadmap(roadmap)
    au = parse_latest_audit(audit)
    times = set(_row_times(au["rows"]))
    checks = []

    def add(port, name, status, note):
        checks.append({"port": port, "check": name, "status": status, "note": note})

    add("Home", "roadmap_lists_1645_audit",
        "pass" if "audit:16:45" in rm["on"] else "fail", "ROADMAP ON has 16:45 daily audit")
    add("FieldCore", "latest_audit_entry_found",
        "pass" if au["date"] else "fail", "newest entry %s" % au["date"])
    audit_row = any("16:45" in " ".join(r[1:3]) and "daily" in " ".join(r).lower()
                    for r in au["rows"])
    add("FieldCore", "latest_entry_has_daily_1645_row",
        "pass" if audit_row else "fail", "daily 16:45 row in newest entry")
    on_times = sorted({t for t in rm["on"] if not t.startswith("audit:")})
    missing = [t for t in on_times if t not in times]
    add("Flagstaff", "roadmap_on_slots_in_audit",
        "pass" if not missing else "fail", "missing=%s" % missing)
    revived = [t for t in rm["paused"] if t in times]
    add("Flagstaff", "paused_slots_not_recreated",
        "pass" if not revived else "fail", "paused=%s revived=%s" % (rm["paused"], revived))
    body = au["body"]
    stated = all(t in body for t in rm["paused"]) and "not scheduled" in body
    add("OSai", "audit_states_paused_not_scheduled",
        "pass" if stated else "fail", "paused line present")
    add("Home", "no_github_dir", "fail" if has_github else "pass", ".github absent")
    extra = sorted(times - set(on_times))
    add("OSai", "audit_only_slots", "advisory" if extra else "pass",
        "in audit, not ROADMAP ON: %s" % extra)
    if rm["raw_weekly"] and "Sun" not in body:
        add("JuniorLLM", "weekly_day_wording", "advisory",
            "ROADMAP says Sun/Mon 17:00, audit says Mon 17:00")
    else:
        add("JuniorLLM", "weekly_day_wording", "pass", "weekly day agrees")
    add("JuniorLLM", "loopback_port_map",
        "pass" if BIND == "127.0.0.1" and LOOPBACK_PORTS == {"bitnetd": 8765, "i2sd": 8767}
        else "fail", "bitnetd %s:8765, i2sd %s:8767" % (BIND, BIND))
    flags = (MINT, SOCKET, FETCH, LAYER_ONE, UE5_LAUNCH, MODEL_PULL)
    add("Home", "path_flags", "pass" if not any(flags) and RATE == 0.05 and OPT_IN
        and ANTICON and POOL_HELD else "fail", "rate 0.05, all gates false")

    prev, chain = GENESIS, []
    for c in checks:
        text = "%s|%s|%s" % (c["port"], c["check"], c["status"])
        prev = close(prev, PACK, text)
        chain.append(prev)
    counts = {k: sum(1 for c in checks if c["status"] == k) for k in ("pass", "advisory", "fail")}
    return {"item": ROADMAP_ITEM, "audit_date": au["date"], "checks": checks,
            "counts": counts, "close": chain[-1] if chain else GENESIS, "chain": chain,
            "ports": list(PORTS), "bind": BIND}


_FIX_RM = """# r\n## Automations (current)\nON:\n- 01:00 overnight\n- 19:00 slot\n- 16:45 daily audit\n- weekly Sun/Mon 17:00\n\nPAUSED:\n- 07:00, 13:00\n\n## Order\n1. x\n"""
_FIX_AU = """# Automation audit 2026-01-01 16:45 MDT\n\n| Name | When | Job |\n|---|---|---|\n| daily audit | daily 16:45 | x |\n\n# Automation audit 2026-01-02 16:45 MDT (tip a)\n\n| Name | Schedule (MDT) | Notes |\n|---|---|---|\n| overnight | daily 01:00 | g |\n| daily audit | daily 16:45 | this |\n| slot 19:00 | daily 19:00 | g |\n| weekly | Mon 17:00 | w |\n| omega | Sat 10:00 | r |\n\n07:00 and 13:00 not scheduled.\n"""


def selftest() -> int:
    assert pack((1, 0, -1, 1)) == bytes([0b01001001])
    assert pack((1,)) == bytes([0b01000000])
    assert close(GENESIS, PACK, "x") == hashlib.sha256(
        GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    r = evaluate(_FIX_RM, _FIX_AU, False)
    assert r["audit_date"] == "2026-01-02", r["audit_date"]
    assert r["counts"]["fail"] == 0, r["checks"]
    assert r["counts"]["advisory"] == 2, r["checks"]
    # paused slot revived -> fail
    bad = _FIX_AU + "| slot 07:00 | daily 07:00 | x |\n"
    assert evaluate(_FIX_RM, bad, False)["counts"]["fail"] >= 1
    # .github present -> fail
    assert evaluate(_FIX_RM, _FIX_AU, True)["counts"]["fail"] == 1
    # audit slot missing from roadmap -> fail
    assert evaluate(_FIX_RM.replace("- 16:45 daily audit\n", ""), _FIX_AU, False)["counts"]["fail"] == 1
    # chain is deterministic and order-sensitive
    a = evaluate(_FIX_RM, _FIX_AU, False)
    assert a["close"] == r["close"] and len(set(a["chain"])) == len(a["chain"])
    print("selftest ok")
    return 0


def main(argv) -> int:
    if "--selftest" in argv:
        selftest()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    try:
        with open(os.path.join(root, "docs", "ROADMAP.md"), encoding="utf-8") as f:
            rm = f.read()
        with open(os.path.join(root, "docs", "AUTOMATION_AUDIT.md"), encoding="utf-8") as f:
            au = f.read()
    except OSError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    res = evaluate(rm, au, os.path.isdir(os.path.join(root, ".github")))
    res.pop("chain")
    print(json.dumps(res, indent=1))
    return 1 if res["counts"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
