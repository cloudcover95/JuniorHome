"""16:45 daily audit, remaining engines port map: BitNet-mlx, web3node, JuniorQuant.

Advances JuniorHome docs/ROADMAP.md named item, Automations (current) ON:
"16:45 daily audit" (writes docs/AUTOMATION_AUDIT.md), remaining engines only.

The audit's standing check is "GitHub Actions stay off" ("GitHub Actions on
JuniorLLM, JuniorStock, BitNet-mlx are already manual-only"; "Do not add CI
yaml"). JuniorLLM, JuniorStock and stocksnode already carry their own 16:45
files. This file maps the three engines still uncovered, from the Home side:

  * ENGINE_SNAPSHOT: a declared, dated read-only observation of each engine's
    .github/workflows at its tip (BitNet-mlx 5df39f9, web3node 75fc5a4,
    JuniorQuant 44496b9; observed 2026-10-23 02:37 MDT). The script never
    fetches; it parses the declared YAML text in-process and checks every
    workflow is workflow_dispatch only with every job gated "if: false".
  * docs/AUTOMATION_AUDIT.md: whether the Actions manual-only line names the
    engine, and whether the newest 16:45 entry still carries an Actions row.
  * docs/REPOS.md and docs/ECOSYSTEM_PORTS.md: each engine has a role row and
    is listed in the ecosystem port list.
Missing coverage in the audit text is an advisory (the audit is written by the
16:45 slot, not by this file); an engine workflow with a push/schedule/PR
trigger or an ungated job is a fail.

It does not import, wrap or re-check any other ports/ file. Each check is
mapped to one port (Home, OSai, FieldCore, Flagstaff, JuniorLLM) and signed
into one chain, in-process, no sockets, no fetch.
Existing path: opt-in, AntiCon, one trit pack, base rate 0.05, pool held,
mint false, socket false, fetch false, layer_one false, ue5_launch false,
bind 127.0.0.1 (bitnetd 8765, i2sd 8767 listed in the port map only).
Close rule: sha256(prev || pack || text)[:16], trits as 2-bit codes
(0->0, 1->1, -1->2), MSB first, zero-padded. Stdlib only.

Usage: python3 ports/daily_audit_remaining_engines_port_map.py [--selftest] [--root DIR]
Exit 0 when no check fails (advisories allowed), 1 otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile

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
ROADMAP_ITEM = "ROADMAP.md Automations ON: 16:45 daily audit (remaining engines)"
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
TRIT_PACK = (1, -1, 0, 1, 1, 0, -1, 1)
_ENC = {0: 0, 1: 1, -1: 2}

# Declared read-only observation (no fetch at run time).
ENGINE_SNAPSHOT = {
    "BitNet-mlx": {
        "tip": "5df39f9",
        "port": "JuniorLLM",
        "ecosystem_name": "BitNet-mlx",
        "workflows": {
            "blackbox_ci.yml": (
                "name: BitNet-MLX Blackbox (manual only)\n"
                "on:\n  workflow_dispatch:\n"
                "jobs:\n  offline_audit:\n    if: false\n    runs-on: macos-latest\n"
            ),
        },
    },
    "web3node": {
        "tip": "75fc5a4",
        "port": "Flagstaff",
        "ecosystem_name": "web3node",
        "workflows": {
            "mdbook.yml": (
                "name: mdBook (manual only)\n"
                "on:\n  workflow_dispatch:\n"
                "jobs:\n  build:\n    if: false\n    runs-on: ubuntu-latest\n"
            ),
        },
    },
    "JuniorQuant": {
        "tip": "44496b9",
        "port": "FieldCore",
        "ecosystem_name": "Quant",
        "workflows": {},
    },
}

_ENTRY = re.compile(r"^# Automation audit (\d{4}-\d{2}-\d{2}) [~]?16:45 MDT.*$", re.M)
_ACTIONS_LINE = re.compile(r"^GitHub Actions on (.+?) are already manual-only\.?\s*$", re.M)


def pack_bytes(trits) -> bytes:
    bits = "".join(format(_ENC[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def parse_workflow(text: str):
    """Return (triggers, jobs_gated, job_count) from a tiny YAML subset."""
    triggers, jobs, gated = [], [], 0
    section = None
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0:
            key, _, rest = line.partition(":")
            section = key.strip().strip("'\"")
            rest = rest.strip()
            if section in ("on", "true") and rest:
                rest = rest.strip("[]")
                triggers += [t.strip() for t in rest.split(",") if t.strip()]
            continue
        if section in ("on", "true") and indent == 2:
            triggers.append(line.split(":")[0].strip().lstrip("- ").strip())
        elif section == "jobs":
            if indent == 2 and line.endswith(":"):
                jobs.append(line[:-1])
            elif indent == 4 and re.fullmatch(r"if:\s*(false|\$\{\{\s*false\s*\}\})", line):
                gated += 1
    return triggers, gated, len(jobs)


def newest_entry(text: str):
    ms = list(_ENTRY.finditer(text))
    if not ms:
        return None, ""
    m = ms[-1]
    return m.group(1), text[m.end():]


def run(root: str, snapshot=None):
    snapshot = ENGINE_SNAPSHOT if snapshot is None else snapshot
    rows = []

    def add(port, name, status, detail):
        rows.append({"port": port, "check": name, "status": status, "detail": detail})

    def read(rel):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return fh.read()

    roadmap = read("docs/ROADMAP.md") or ""
    audit = read("docs/AUTOMATION_AUDIT.md") or ""
    repos = read("docs/REPOS.md") or ""
    eco = read("docs/ECOSYSTEM_PORTS.md") or ""

    on_block = roadmap.split("ON:", 1)[1].split("PAUSED:", 1)[0] if "ON:" in roadmap else ""
    add("Home", "roadmap ON lists 16:45 daily audit",
        "pass" if "16:45 daily audit" in on_block else "fail", "docs/ROADMAP.md")

    m = _ACTIONS_LINE.search(audit)
    named = [n.strip() for n in re.split(r",|\band\b", m.group(1))] if m else []
    named = [n for n in named if n]
    add("Home", "audit Actions manual-only line present",
        "pass" if m else "advisory", ", ".join(named) or "line missing")

    date, body = newest_entry(audit)
    add("Home", "newest 16:45 audit entry found", "pass" if date else "fail", date or "none")
    has_row = bool(re.search(r"GitHub Actions|Actions stay off|CI yaml", body))
    add("Home", "newest entry carries an Actions row",
        "pass" if has_row else "advisory",
        "present" if has_row else f"{date}: no Actions row (dropped since earlier entries)")

    for eng, snap in snapshot.items():
        port = snap["port"]
        add(port, f"{eng}: named in audit Actions line",
            "pass" if eng in named else "advisory",
            "named" if eng in named else "not named; audit does not cover this engine")
        role = re.search(r"^\|[^|\n]*\b" + re.escape(eng) + r"\b[^|\n]*\|\s*([^|\n]+?)\s*\|", repos, re.M)
        add(port, f"{eng}: REPOS.md role row", "pass" if role else "fail",
            role.group(1) if role else "missing")
        add(port, f"{eng}: in ECOSYSTEM_PORTS list",
            "pass" if re.search(r"\b" + re.escape(snap["ecosystem_name"]) + r"\b", eco) else "fail",
            snap["ecosystem_name"])
        wfs = snap["workflows"]
        if not wfs:
            add(port, f"{eng}@{snap['tip']}: no workflows", "pass", "no .github/workflows")
        for wf, text in sorted(wfs.items()):
            trig, gated, njobs = parse_workflow(text)
            ok_trig = bool(trig) and set(trig) == {"workflow_dispatch"}
            add(port, f"{eng}@{snap['tip']}: {wf} manual-only trigger",
                "pass" if ok_trig else "fail", ",".join(trig) or "none")
            add(port, f"{eng}@{snap['tip']}: {wf} jobs gated if:false",
                "pass" if njobs and gated == njobs else "fail", f"{gated}/{njobs}")

    flags_ok = (not MINT and not SOCKET and not FETCH and not LAYER_ONE and not UE5_LAUNCH
                and not MODEL_PULL and BIND == "127.0.0.1" and RATE == 0.05)
    add("OSai", "path flags (mint/socket/fetch/layer_one/ue5_launch false, rate 0.05)",
        "pass" if flags_ok else "fail", f"bind {BIND} {LOOPBACK_PORTS}")
    add("JuniorLLM", "no .github in JuniorHome",
        "pass" if not os.path.isdir(os.path.join(root, ".github")) else "fail", "no CI yaml")

    pack = pack_bytes(TRIT_PACK)
    prev = GENESIS
    for r in rows:
        text = json.dumps(r, sort_keys=True)
        prev = close(prev, pack, text)
        r["close"] = prev
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("pass", "advisory", "fail")}
    return rows, counts, prev


def _selftest() -> int:
    assert pack_bytes((1, -1, 0, 1)) == bytes([0b01100001])
    assert close(GENESIS, b"\x00", "x") == hashlib.sha256(b"0" * 16 + b"\x00x").hexdigest()[:16]
    t, g, n = parse_workflow("on:\n  workflow_dispatch:\njobs:\n  a:\n    if: false\n")
    assert (t, g, n) == (["workflow_dispatch"], 1, 1), (t, g, n)
    t, g, n = parse_workflow("on: [push, workflow_dispatch]\njobs:\n  a:\n    runs-on: x\n")
    assert t == ["push", "workflow_dispatch"] and g == 0 and n == 1, (t, g, n)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "docs"))
        files = {
            "ROADMAP.md": "ON:\n- 16:45 daily audit\nPAUSED:\n- 07:00\n",
            "AUTOMATION_AUDIT.md": ("GitHub Actions on JuniorLLM, BitNet-mlx are already manual-only.\n"
                                    "# Automation audit 2026-10-22 16:45 MDT (tip x)\n| a | b |\n"),
            "REPOS.md": "| BitNet-mlx | MLX |\n| web3node / JuniorSOL | chain |\n| JuniorStock / JuniorQuant | book |\n",
            "ECOSYSTEM_PORTS.md": "BitNet-mlx, web3node, Quant\n",
        }
        for k, v in files.items():
            with open(os.path.join(d, "docs", k), "w", encoding="utf-8") as fh:
                fh.write(v)
        rows, c, last = run(d)
        assert c["fail"] == 0, rows
        assert c["advisory"] == 3, c  # no Actions row, web3node + JuniorQuant unnamed
        rows2, _, last2 = run(d)
        assert last == last2 and len(last) == 16
        bad = json.loads(json.dumps(ENGINE_SNAPSHOT))
        bad["web3node"]["workflows"]["mdbook.yml"] = "on:\n  push:\njobs:\n  b:\n    runs-on: x\n"
        _, c3, _ = run(d, bad)
        assert c3["fail"] == 2, c3
    print("selftest ok")
    return 0


def main(argv) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--root" in argv:
        root = argv[argv.index("--root") + 1]
    if "--selftest" in argv:
        _selftest()
    rows, counts, last = run(root)
    for r in rows:
        print(f"[{r['status']:8}] {r['port']:9} {r['check']} :: {r['detail']}")
    print(json.dumps({"item": ROADMAP_ITEM, "counts": counts, "close": last,
                      "rule": CLOSE, "rate": RATE}, sort_keys=True))
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
