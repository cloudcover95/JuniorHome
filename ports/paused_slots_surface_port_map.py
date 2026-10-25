"""Paused slots surface port map: ROADMAP PAUSED 07:00, 13:00 vs every doc
surface that names those two slots.

Advances JuniorHome docs/ROADMAP.md named item, Automations (current):
    PAUSED:
    - 07:00, 13:00
    Do not recreate paused slots.

New angle. The existing paused-slot advances (scripts/advance_paused_*,
web3node/advance_paused_slots*) hold declarative facts and read no file;
ports/daily_audit_slot_port_map.py checks only the newest 16:45 entry in
docs/AUTOMATION_AUDIT.md; ports/chief_weekly_receipt_port_map.py checks only
the Friday weekly 7716afb1 line and receipt headers. None reads the chief
weekly's live catalog tables, its Gaps/Skipped sections, docs/JUNIOROS.md,
docs/OS_MESH.md or the slots_paused literal in web3node/probe_future.py.
This file reads those surfaces (read-only, located relative to this
script) and does not import, wrap or re-check any of the files above.

  Home      ROADMAP has "- 07:00, 13:00" under PAUSED, the note
            "Do not recreate paused slots." verbatim, and no ON line
            carries a paused time.
  Flagstaff docs/WEEKLY_AUDIT.md (chief weekly, read only): no live
            catalog row is scheduled at a paused time; the newest section
            states they are not in the catalog; any "recreate 07:00/13:00"
            line sits under "## Skipped" only. Advisory when a section
            grades a paused slot FAIL or lists it under Gaps as missing
            capacity, since ROADMAP calls it PAUSED, not missing.
  OSai      docs/JUNIOROS.md names both slots as not scheduled and its
            Automations line has no paused time.
  FieldCore docs/OS_MESH.md names both slots as not scheduled.
  JuniorLLM web3node/probe_future.py slots_paused literal (parsed with ast,
            never imported or run) equals the ROADMAP PAUSED set; loopback
            port map bitnetd 127.0.0.1:8765, i2sd 127.0.0.1:8767.

Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false, fetch false, layer_one false, ue5_launch false,
bind 127.0.0.1. No socket opened, no network, no model pull, no UE boot,
no poll sets the rate. Close rule: sha256(prev || pack || text)[:16],
balanced trits packed as 2-bit codes (0->0, 1->1, -1->2), MSB first,
zero-padded. Stdlib only.

Usage: python3 ports/paused_slots_surface_port_map.py [--selftest] [--root DIR]
Exit 0 when no check fails (advisories allowed), 1 otherwise.
"""
from __future__ import annotations

import ast
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
ITEM = "PAUSED: - 07:00, 13:00 / Do not recreate paused slots."
ITEM_LINE = "- 07:00, 13:00"
ITEM_NOTE = "Do not recreate paused slots."
ROADMAP_ITEM = "ROADMAP.md Automations (current) " + ITEM
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
TRIT_PACK = (1, -1, 0, 1, 1, 0, -1, 1)
_ENC = {0: 0, 1: 1, -1: 2}

SURFACES = {
    "roadmap": "docs/ROADMAP.md",
    "weekly": "docs/WEEKLY_AUDIT.md",
    "junioros": "docs/JUNIOROS.md",
    "os_mesh": "docs/OS_MESH.md",
    "probe": "web3node/probe_future.py",
}

_TIME = re.compile(r"\b([0-2]\d:[0-5]\d)\b")
_WEEKLY = re.compile(r"^# Weekly audit (\d{4}-\d{2}-\d{2}) ([0-2]\d:[0-5]\d) MDT.*$", re.M)
_SHORT = re.compile(r"\b07\s*/\s*13\b")


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
    on, paused, paused_lines, mode, in_block = [], [], [], None, False
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
        elif s.startswith("-") and mode == "on":
            on.append(s)
        elif s.startswith("-") and mode == "paused":
            paused_lines.append(s)
            if "weekly" not in s.lower():
                paused.extend(_TIME.findall(s))
    return {"on": on, "paused": paused, "paused_lines": paused_lines,
            "note": ITEM_NOTE in text}


def _mentions_paused(s: str, paused) -> bool:
    return any(t in s for t in paused) or bool(_SHORT.search(s))


def _snip(s: str, paused, width: int = 60) -> str:
    s = s.strip()
    idx = [s.find(t) for t in paused if t in s]
    m = _SHORT.search(s)
    if m:
        idx.append(m.start())
    i = max(0, min(idx) - 10) if idx else 0
    return s[i:i + width]


def weekly_sections(text: str) -> list:
    heads = list(_WEEKLY.finditer(text))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[h.end():end]
        subs, cur = {}, "_top"
        for line in body.splitlines():
            if line.startswith("## "):
                cur = line[3:].strip()
                subs.setdefault(cur, [])
                continue
            subs.setdefault(cur, []).append(line)
        out.append({"date": h.group(1), "time": h.group(2), "subs": subs})
    return out


def _cells(line: str):
    s = line.strip()
    if not s.startswith("|") or set(s) <= set("|-: "):
        return None
    return [c.strip() for c in s.strip("|").split("|")]


def catalog_schedules(lines) -> list:
    """Schedule-column cells from a live-list table (not nextRun stamps)."""
    col, out = None, []
    for line in lines:
        cells = _cells(line)
        if cells is None:
            continue
        low = [c.lower() for c in cells]
        if "schedule" in low:
            col = low.index("schedule")
            continue
        if col is not None and col < len(cells):
            out.append((cells[0], cells[col]))
    return out


def scan_weekly(text: str, paused) -> dict:
    secs = weekly_sections(text)
    live_hits, fail_grades, gap_lines, recreate_bad = [], [], [], []
    for sec in secs:
        for name, lines in sec["subs"].items():
            nm = name.lower()
            if nm.startswith("automations (live"):
                for row, sched in catalog_schedules(lines):
                    if any(t in sched for t in paused):
                        live_hits.append("%s %s=%s" % (sec["date"], row, sched))
            for line in lines:
                cells = _cells(line)
                if cells and _mentions_paused(cells[0], paused) and \
                        any("FAIL" in c for c in cells[1:]):
                    fail_grades.append("%s %s" % (sec["date"], cells[0]))
                if nm == "gaps" and _mentions_paused(line, paused):
                    gap_lines.append("%s ...%s" % (sec["date"], _snip(line, paused)))
                if "recreat" in line.lower() and _mentions_paused(line, paused) \
                        and nm != "skipped":
                    recreate_bad.append("%s [%s] %s" % (sec["date"], name, line.strip()[:60]))
    newest = secs[-1] if secs else None
    stated = False
    if newest:
        blob = "\n".join("\n".join(v) for v in newest["subs"].values())
        stated = all(t in blob for t in paused) and bool(
            re.search(r"not in (the )?(live )?(list|catalog)|no 07:00 or 13:00", blob, re.I))
    return {"sections": [s["date"] for s in secs], "newest": newest["date"] if newest else None,
            "live_hits": live_hits, "fail_grades": fail_grades, "gap_lines": gap_lines,
            "recreate_bad": recreate_bad, "newest_states": stated}


def probe_slots(src: str):
    """Return the slots_paused list literal from probe_future.py via ast."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "slots_paused":
                    try:
                        return list(ast.literal_eval(v))
                    except ValueError:
                        return None
    return None


def _doc_not_scheduled(text: str, paused) -> bool:
    for line in text.splitlines():
        if all(t in line for t in paused) and "not scheduled" in line:
            return True
    return False


def evaluate(docs: dict, has_github: bool) -> dict:
    rm = parse_roadmap(docs.get("roadmap", ""))
    paused = rm["paused"]
    checks = []

    def add(port, name, status, note):
        checks.append({"port": port, "check": name, "status": status, "note": note})

    add("Home", "roadmap_paused_line_verbatim",
        "pass" if ITEM_LINE in rm["paused_lines"] and paused == ["07:00", "13:00"] else "fail",
        "PAUSED times=%s" % paused)
    add("Home", "roadmap_note_verbatim", "pass" if rm["note"] else "fail", ITEM_NOTE)
    on_hits = [s for s in rm["on"] if any(t in _TIME.findall(s) for t in paused)]
    add("Home", "roadmap_on_excludes_paused", "pass" if not on_hits else "fail",
        "ON lines with a paused time: %s" % on_hits)

    w = scan_weekly(docs.get("weekly", ""), paused)
    add("Flagstaff", "weekly_sections_found", "pass" if w["sections"] else "fail",
        "sections=%s" % w["sections"])
    add("Flagstaff", "weekly_live_catalog_excludes_paused",
        "pass" if w["sections"] and not w["live_hits"] else "fail",
        "schedule cells at a paused time: %s" % w["live_hits"])
    add("Flagstaff", "weekly_newest_states_not_in_catalog",
        "pass" if w["newest_states"] else "advisory", "newest=%s" % w["newest"])
    add("Flagstaff", "weekly_recreate_only_under_skipped",
        "pass" if not w["recreate_bad"] else "fail", "outside Skipped: %s" % w["recreate_bad"])
    add("Flagstaff", "weekly_grades_paused_as_fail",
        "advisory" if w["fail_grades"] else "pass",
        "paused slot graded FAIL (ROADMAP says PAUSED): %s" % w["fail_grades"])
    add("Flagstaff", "weekly_gaps_list_paused",
        "advisory" if w["gap_lines"] else "pass",
        "paused slot listed as a capacity gap: %s" % w["gap_lines"])

    jo = docs.get("junioros", "")
    add("OSai", "junioros_paused_not_scheduled",
        "pass" if _doc_not_scheduled(jo, paused) else "fail", SURFACES["junioros"])
    auto = [l for l in jo.splitlines() if l.lower().startswith("automations")]
    jo_hits = [l for l in auto if any(t in l for t in paused)]
    add("OSai", "junioros_automations_excludes_paused",
        "pass" if auto and not jo_hits else ("fail" if jo_hits else "advisory"),
        "automations lines=%d paused hits=%d" % (len(auto), len(jo_hits)))

    add("FieldCore", "os_mesh_paused_not_scheduled",
        "pass" if _doc_not_scheduled(docs.get("os_mesh", ""), paused) else "fail",
        SURFACES["os_mesh"])

    ps = probe_slots(docs.get("probe", ""))
    add("JuniorLLM", "probe_future_slots_paused_match",
        "pass" if ps is not None and sorted(ps) == sorted(paused) else "fail",
        "slots_paused=%s roadmap=%s (ast, not imported)" % (ps, paused))
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
    return {"item": ROADMAP_ITEM, "paused": paused, "weekly_sections": w["sections"],
            "checks": checks, "counts": counts, "close": chain[-1] if chain else GENESIS,
            "chain": chain, "ports": list(PORTS), "bind": BIND}


_FIX = {
    "roadmap": ("# r\n## Automations (current)\nON:\n- 01:00 overnight\n- 19:00 slot\n"
                "- 16:45 daily audit\n- weekly Sun/Mon 17:00\n\nPAUSED:\n- 07:00, 13:00\n"
                "- Friday weekly `7716afb1` (merged into chief)\n\n"
                "Do not recreate paused slots.\n\n## Order\n1. x\n"),
    "weekly": ("# Weekly audit 2026-01-05 17:04 MDT\n\n## Automations (live list)\n\n"
               "| name | id | schedule | nextRun |\n|---|---|---|---|\n"
               "| overnight | a | daily 01:00 Denver | 2026-01-05T07:00Z |\n\n"
               "Not in live list: slot 07:00, slot 13:00.\n\n## Skipped\n\n"
               "- Recreate 07:00 / 13:00 automations (audit does not create tasks).\n\n"
               "## Gaps\n\n1. llama GGUF not on box.\n"),
    "junioros": ("# JuniorOS\nAutomations: daily 16:45 audit, 01:00 and 19:00 slices.\n"
                 "07:00 and 13:00 are named in the daily prompt and are not scheduled.\n"),
    "os_mesh": "Bind 127.0.0.1.\n07:00 and 13:00 are not scheduled.\n",
    "probe": 'def probe():\n    return {"t_next": "T13", "slots_paused": ["07:00", "13:00"]}\n',
}


def _ev(**over):
    d = dict(_FIX)
    d.update(over)
    return evaluate(d, False)


def selftest() -> int:
    assert pack((1, 0, -1, 1)) == bytes([0b01001001])
    assert pack((1,)) == bytes([0b01000000])
    assert len(PACK) == 2
    assert close(GENESIS, PACK, "x") == hashlib.sha256(
        GENESIS.encode() + PACK + b"x").hexdigest()[:16]
    r = evaluate(dict(_FIX), False)
    assert r["paused"] == ["07:00", "13:00"], r["paused"]
    assert r["counts"] == {"pass": len(r["checks"]), "advisory": 0, "fail": 0}, r["checks"]
    # nextRun stamp at 07:00Z is not a schedule -> still pass
    assert not [c for c in r["checks"] if c["check"] == "weekly_live_catalog_excludes_paused"
                and c["status"] != "pass"]
    # paused time moved into ON -> fail
    bad_rm = _FIX["roadmap"].replace("- 19:00 slot\n", "- 19:00 slot\n- 07:00 slot\n")
    assert _ev(roadmap=bad_rm)["counts"]["fail"] >= 1
    # PAUSED line or note missing -> fail
    assert _ev(roadmap=_FIX["roadmap"].replace("- 07:00, 13:00\n", ""))["counts"]["fail"] >= 1
    assert _ev(roadmap=_FIX["roadmap"].replace(ITEM_NOTE, ""))["counts"]["fail"] == 1
    # live catalog row scheduled at a paused slot -> fail
    w_live = _FIX["weekly"].replace("| overnight |", "| slot 07:00 | b | daily 07:00 Denver | - |\n| overnight |")
    assert _ev(weekly=w_live)["counts"]["fail"] == 1
    # recreate line outside Skipped -> fail
    w_rec = _FIX["weekly"] + "\n## Next\n\n- Recreate 07:00 and 13:00 slots.\n"
    assert _ev(weekly=w_rec)["counts"]["fail"] == 1
    # paused graded FAIL / listed as gap -> advisories, never fails
    w_adv = _FIX["weekly"].replace("1. llama", "2. 07:00 and 13:00 slots absent from catalog.\n1. llama")
    w_adv += "\n## Slots\n\n| slot | last 7 days |\n|---|---|\n| 07:00 | FAIL - not listed |\n"
    ra = _ev(weekly=w_adv)
    assert ra["counts"]["fail"] == 0 and ra["counts"]["advisory"] == 2, ra["checks"]
    # probe literal drift -> fail; probe not importable is irrelevant (ast only)
    assert _ev(probe=_FIX["probe"].replace('"13:00"', '"12:00"'))["counts"]["fail"] == 1
    assert _ev(probe="import nonexistent_mod\nX = {'slots_paused': ['07:00', '13:00']}\n")[
        "counts"]["fail"] == 0
    # surface docs missing the not-scheduled line -> fail
    assert _ev(os_mesh="Bind 127.0.0.1.\n")["counts"]["fail"] == 1
    assert _ev(junioros="# JuniorOS\nAutomations: daily 16:45 audit.\n")["counts"]["fail"] == 1
    # .github present -> fail
    assert evaluate(dict(_FIX), True)["counts"]["fail"] == 1
    # chain deterministic and order-sensitive
    again = evaluate(dict(_FIX), False)
    assert again["close"] == r["close"] and len(set(again["chain"])) == len(again["chain"])
    assert "socket" not in sys.modules or not SOCKET
    print("selftest ok")
    return 0


def _read(root: str, rel: str) -> str:
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return f.read()


def main(argv) -> int:
    if "--selftest" in argv:
        selftest()
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    try:
        docs = {k: _read(root, v) for k, v in SURFACES.items()}
    except OSError as e:
        print(json.dumps({"error": str(e)}))
        return 1
    res = evaluate(docs, os.path.isdir(os.path.join(root, ".github")))
    res.pop("chain")
    res["surfaces"] = SURFACES
    print(json.dumps(res, indent=1))
    c = res["counts"]
    print("pass=%d advisory=%d fail=%d close=%s" % (c["pass"], c["advisory"], c["fail"], res["close"]))
    return 1 if c["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
