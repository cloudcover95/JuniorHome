#!/usr/bin/env python3
"""i2sd userspace doc port registry: who owns 127.0.0.1:8767 across Home docs.

ROADMAP item (JuniorHome docs/ROADMAP.md, Order #3, verbatim):
    3. **i2sd userspace** — 127.0.0.1:8767. No docker.sock.

New angle: the other i2sd ports/ files hold declarative facts or a hardcoded
port ledger. This file reads the Home docs that actually publish the loopback
map (README.md and the top-level docs/*.md, read-only, located relative to
this script) and builds the registry those docs imply, then checks that the
docs agree with ROADMAP #3:

  * Home      ROADMAP #3 is present verbatim and sits between #2 GGUF and
              #4 Asahi MLX (order, do not skip).
  * OSai      the bind lists in docs/TECH_AUDIT.md and docs/ZERO_TRUST.md
              both name i2sd under 127.0.0.1.
  * FieldCore the docs/SUITE_GLUE.md endpoint row "i2sd | 127.0.0.1:8767"
              exists and every Home doc it cites exists and names i2sd 8767.
  * Flagstaff every service-labelled port in the docs maps i2sd -> 8767 only
              and 8767 -> i2sd only; no 0.0.0.0 on 8767; every docker.sock
              mention is a deny line.
  * JuniorLLM the rails/linux lane (docs/INTEGRATION.md) and the JuniorOS
              overlay stack (docs/JUNIOROS_KERNEL.md) both carry i2sd.

Unlabelled 8767 mentions, bare ":8767" without a host, and neighbour-port
sharing (8765 named for both bitnetd and another surface) are advisories.
Ports touched (in-process map only, never opened): i2sd 127.0.0.1:8767;
bitnetd 127.0.0.1:8765, deck/app 127.0.0.1:8766, hook 127.0.0.1:8770 and
UI 127.0.0.1:8771 are read as neighbours for collision checks.

It does not import, wrap, re-check or index any other ports/, scripts/ or
web3node/ file. No sockets, no network, no fetch, no subprocess, no model
pull, no UE boot, no writes. Stdlib only. Base rate 0.05 (a constant), pool
held, mint/socket/layer_one/fetch/ue5_launch false, bind 127.0.0.1.
One trit pack (pass=1, advisory=0, fail=-1, 2-bit i2s codes 0->0, 1->1,
-1->2, MSB first, zero-padded). Each row closes with
sha256(prev || pack || text)[:16] hex, starting from 16 zeros.

Usage: python3 ports/i2sd_userspace_doc_port_registry.py [--selftest]
Exit 0 when no row fails (advisories allowed), 1 otherwise.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

RATE = 0.05
OPT_IN = True
ANTICON = True
POOL_HELD = True
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MODEL_PULL = False
DOCKER_SOCK = False
BIND = "127.0.0.1"
I2SD_PORT = 8767
I2SD = f"{BIND}:{I2SD_PORT}"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
CLOSE_RULE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROADMAP_ITEM = "3. **i2sd userspace** — 127.0.0.1:8767. No docker.sock."
ROADMAP_PREV = "2. **GGUF**"
ROADMAP_NEXT = "4. **Asahi MLX**"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Service labels as the Home docs write them -> canonical name.
_LABELS = {
    "i2sd": "i2sd", "hook": "hook", "ui": "ui", "bitnetd": "bitnetd",
    "legacy_desk": "deck", "deck": "deck", "app": "deck",
}
_LBL = r"(i2sd|hook|UI|bitnetd|legacy_desk|deck)"
_HOST = r"(?:127\.0\.0\.1|0\.0\.0\.0)?"
# "i2sd :8767", "i2sd | 127.0.0.1:8767", "hook 127.0.0.1:8770"
_LABEL_THEN_PORT = re.compile(_LBL + r"[\s|`(]{0,6}" + _HOST + r":(\d{4})")
# "8767 i2sd", "8771 UI" -- a bare number only (not ":8770"), and the label
# must not itself be followed by a port ("Hook :8770  i2sd :8767" is two
# label-then-port pairs, not "8770 i2sd").
_PORT_THEN_LABEL = re.compile(r"(?<![\d.:])(\d{4})\s+" + _LBL + r"\b"
                              r"(?![\s|`(]{0,6}" + _HOST + r":\d{4})")
_DENY = re.compile(r"\b(no|not|never|any|deny|denied|rejects?|without)\b", re.I)

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {v: k for k, v in _ENC.items()}
_TRIT = {"pass": 1, "advisory": 0, "fail": -1}


def pack(trits) -> bytes:
    """Balanced trits as 2-bit codes, MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack(data: bytes, n: int):
    trits = []
    for byte in data:
        for shift in (6, 4, 2, 0):
            if len(trits) < n:
                trits.append(_DEC[(byte >> shift) & 3])
    return tuple(trits)


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def load_docs(root: str = ROOT) -> dict:
    """README.md + top-level docs/*.md, read-only. Nothing else is opened."""
    docs = {}
    paths = ["README.md"]
    ddir = os.path.join(root, "docs")
    if os.path.isdir(ddir):
        paths += ["docs/" + n for n in sorted(os.listdir(ddir)) if n.endswith(".md")]
    for rel in paths:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as fh:
                docs[rel] = fh.read()
    return docs


def registry(docs: dict):
    """service -> {port: [(doc, line_no)]}, built from labelled mentions."""
    reg = {}
    for rel, text in docs.items():
        for no, line in enumerate(text.splitlines(), 1):
            hits = [(m.group(1), m.group(2)) for m in _LABEL_THEN_PORT.finditer(line)]
            hits += [(m.group(2), m.group(1)) for m in _PORT_THEN_LABEL.finditer(line)]
            for lbl, port in hits:
                name = _LABELS[lbl.lower()]
                reg.setdefault(name, {}).setdefault(int(port), []).append((rel, no))
    return reg


def checks(docs: dict):
    rows = []

    def add(port, text, verdict):
        rows.append((port, verdict, text))

    # --- Flags (Home)
    flags_ok = (RATE == 0.05 and POOL_HELD and not (MINT or SOCKET or LAYER_ONE
                or FETCH or UE5_LAUNCH or MODEL_PULL or DOCKER_SOCK)
                and BIND == "127.0.0.1")
    add("Home", "flags held: rate 0.05, pool held, mint/socket/layer_one/fetch/"
        "ue5_launch false, bind 127.0.0.1", "pass" if flags_ok else "fail")

    # --- Home: ROADMAP #3 verbatim and in order
    road = docs.get("docs/ROADMAP.md", "")
    lines = [ln.strip() for ln in road.splitlines()]
    if ROADMAP_ITEM in lines:
        i = lines.index(ROADMAP_ITEM)
        prev_ok = any(ln.startswith(ROADMAP_PREV) for ln in lines[:i])
        next_ok = any(ln.startswith(ROADMAP_NEXT) for ln in lines[i + 1:])
        add("Home", "ROADMAP #3 verbatim: " + ROADMAP_ITEM, "pass")
        add("Home", "ROADMAP order: #2 GGUF before #3 i2sd, #4 Asahi MLX after",
            "pass" if prev_ok and next_ok else "fail")
    else:
        add("Home", "ROADMAP #3 verbatim line missing: " + ROADMAP_ITEM, "fail")

    # --- OSai: bind lists name i2sd under 127.0.0.1
    for rel, needle in (("docs/TECH_AUDIT.md", "Bind:"), ("docs/ZERO_TRUST.md", "Bind 127.0.0.1")):
        text = docs.get(rel)
        if text is None:
            add("OSai", f"{rel} missing (bind list not published)", "advisory")
            continue
        hit = [ln for ln in text.splitlines() if needle in ln and "i2sd" in ln]
        good = [ln for ln in hit if "127.0.0.1" in ln and "0.0.0.0" not in ln.split("Not")[0]]
        add("OSai", f"{rel} bind list names i2sd on 127.0.0.1",
            "pass" if good else "fail")

    # --- FieldCore: SUITE_GLUE endpoint row + cited docs
    glue = docs.get("docs/SUITE_GLUE.md", "")
    row = next((ln for ln in glue.splitlines()
                if re.match(r"\|\s*i2sd\s*\|\s*" + re.escape(I2SD) + r"\s*\|", ln)), None)
    if row is None:
        add("FieldCore", "SUITE_GLUE endpoint row 'i2sd | 127.0.0.1:8767' missing", "fail")
    else:
        add("FieldCore", "SUITE_GLUE endpoint row 'i2sd | 127.0.0.1:8767'", "pass")
        for cited in re.findall(r"`(docs/[^`]+\.md)`", row):
            text = docs.get(cited)
            if text is None:
                add("FieldCore", f"SUITE_GLUE cites {cited}: file missing", "fail")
            elif re.search(r"i2sd[^\n]*8767", text):
                add("FieldCore", f"SUITE_GLUE cites {cited}: names i2sd 8767", "pass")
            else:
                add("FieldCore", f"SUITE_GLUE cites {cited}: no i2sd 8767 line", "fail")

    # --- Flagstaff: registry, 0.0.0.0, docker.sock
    reg = registry(docs)
    i2sd_ports = reg.get("i2sd", {})
    wrong = {p: locs for p, locs in i2sd_ports.items() if p != I2SD_PORT}
    if not i2sd_ports:
        add("Flagstaff", "registry: no labelled i2sd port in Home docs", "fail")
    elif wrong:
        add("Flagstaff", "registry: i2sd labelled on other port(s) " +
            ", ".join(f"{p}@{l[0][0]}:{l[0][1]}" for p, l in sorted(wrong.items())), "fail")
    else:
        n = sum(len(v) for v in i2sd_ports.values())
        add("Flagstaff", f"registry: i2sd -> 8767 only ({n} labelled mentions)", "pass")
    squat = {s: ports[I2SD_PORT] for s, ports in reg.items()
             if s != "i2sd" and I2SD_PORT in ports}
    if squat:
        add("Flagstaff", "registry: 8767 also labelled for " +
            ", ".join(f"{s}@{l[0][0]}:{l[0][1]}" for s, l in sorted(squat.items())), "fail")
    else:
        add("Flagstaff", "registry: 8767 -> i2sd only", "pass")

    pub, unlabelled, bare, sock_bad = [], [], [], []
    for rel, text in docs.items():
        for no, line in enumerate(text.splitlines(), 1):
            if "0.0.0.0:8767" in line or ("8767" in line and "0.0.0.0" in line
                                          and not _DENY.search(line)):
                pub.append(f"{rel}:{no}")
            if "8767" in line and "i2sd" not in line:
                unlabelled.append(f"{rel}:{no}")
            if re.search(r"(?<![\d.])(?<!127\.0\.0\.1):8767", line) and "127.0.0.1:8767" not in line:
                bare.append(f"{rel}:{no}")
            if "docker.sock" in line and not _DENY.search(line):
                sock_bad.append(f"{rel}:{no}")
    add("Flagstaff", "no 0.0.0.0 bind on 8767" + (": " + ", ".join(pub) if pub else ""),
        "fail" if pub else "pass")
    add("Flagstaff", "docker.sock only in deny lines" +
        (": " + ", ".join(sock_bad) if sock_bad else ""), "fail" if sock_bad else "pass")
    if unlabelled:
        add("Flagstaff", "8767 named without the i2sd label: " + ", ".join(unlabelled), "advisory")
    if bare:
        add("FieldCore", "bare ':8767' without host 127.0.0.1: " + ", ".join(bare), "advisory")

    # Neighbour sharing (not this item's port; advisory only).
    extra = sorted({f"{rel}:{no}" for s, ports in reg.items() if s != "bitnetd"
                    for rel, no in ports.get(8765, [])})
    for rel, text in docs.items():
        for no, line in enumerate(text.splitlines(), 1):
            loc = f"{rel}:{no}"
            if re.search(r"serve loopback `?:8765", line) and loc not in extra:
                extra.append(loc)
    if extra:
        add("JuniorLLM", "neighbour 8765 (bitnetd) also named for another surface: " +
            ", ".join(extra), "advisory")

    # --- JuniorLLM: rails/linux lane + JuniorOS overlay stack carry i2sd
    integ = docs.get("docs/INTEGRATION.md", "")
    add("JuniorLLM", "INTEGRATION rails/linux lane carries i2sd :8767",
        "pass" if re.search(r"rails/linux\s+i2sd\s+:8767", integ) else "fail")
    kern = docs.get("docs/JUNIOROS_KERNEL.md")
    if kern is None:
        add("JuniorLLM", "docs/JUNIOROS_KERNEL.md missing (overlay stack)", "advisory")
    else:
        ok = any("overlay" in ln and "i2sd" in ln for ln in kern.splitlines())
        add("JuniorLLM", "JuniorOS overlay stack lists i2sd", "pass" if ok else "fail")
    return rows


def run(docs: dict, out=print):
    rows = checks(docs)
    trits = tuple(_TRIT[v] for _, v, _ in rows)
    pk = pack(trits)
    prev = GENESIS
    for port, verdict, text in rows:
        prev = close(prev, pk, f"{port}|{verdict}|{text}")
        out(f"[{verdict:8}] {port:9} {text}  close={prev}")
    counts = {k: sum(1 for _, v, _ in rows if v == k) for k in ("pass", "advisory", "fail")}
    out(f"item: {ROADMAP_ITEM}")
    out(f"ports: i2sd {I2SD} (neighbours 8765 bitnetd, 8766 deck, 8770 hook, "
        f"8771 UI read only); handshake {' -> '.join(PORTS)}; socket={SOCKET}")
    out(f"pass={counts['pass']} advisory={counts['advisory']} fail={counts['fail']} close={prev}")
    return counts, prev


_GOOD = {
    "README.md": "serve loopback 127.0.0.1:8771.\n",
    "docs/ROADMAP.md": ("## Order (do not skip)\n1. **T13+** — x\n2. **GGUF** — y\n"
                        + ROADMAP_ITEM + "\n4. **Asahi MLX** — probe only.\n"),
    "docs/TECH_AUDIT.md": "- Bind: 127.0.0.1:8770 hook / 8771 UI / 8767 i2sd. Not 0.0.0.0.\n",
    "docs/ZERO_TRUST.md": "- Bind 127.0.0.1 only on hook/UI/i2sd\n- No docker.sock\n",
    "docs/SUITE_GLUE.md": "| i2sd | 127.0.0.1:8767 | `docs/OBSIDIAN.md` |\n",
    "docs/OBSIDIAN.md": "i2sd PACK is the IoT path: 127.0.0.1:8767.\n",
    "docs/INTEGRATION.md": "   rails/linux  i2sd :8767   juniorctl\n",
    "docs/JUNIOROS_KERNEL.md": "    overlay (os-release.junior, bitnetd loopback, i2sd)\n",
}


def selftest() -> int:
    ok = True

    def expect(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("selftest FAIL:", msg)

    for n in range(0, 11):
        t = tuple((-1, 0, 1)[(i * 7 + n) % 3] for i in range(n))
        expect(unpack(pack(t), n) == t, f"pack round-trip n={n}")
    expect(pack((1, 0, -1, 1)) == bytes([0b01001001]), "pack bits")
    expect(close(GENESIS, b"\x01", "a") == close(GENESIS, b"\x01", "a"), "close deterministic")
    expect(len(close(GENESIS, b"", "")) == 16, "close length")
    expect(BIND != "0.0.0.0" and not SOCKET and not MINT and RATE == 0.05, "flags")

    sink = []
    c, tip = run(dict(_GOOD), sink.append)
    expect(c["fail"] == 0, f"good fixture has fails: {sink}")
    c2, tip2 = run(dict(_GOOD), [].append)
    expect(tip == tip2, "close reproducible")

    bad_cases = {
        "i2sd moved": ("docs/OBSIDIAN.md", "i2sd PACK is the IoT path: 127.0.0.1:8768.\n"),
        "8767 squatted": ("docs/UI.md", "http://127.0.0.1:8767 hook 127.0.0.1:8767\n"),
        "public bind": ("docs/UI.md", "listen 0.0.0.0:8767\n"),
        "docker.sock mount": ("docs/UI.md", "mount /var/run/docker.sock into i2sd\n"),
        "roadmap edited": ("docs/ROADMAP.md", "3. **i2sd userspace** — 0.0.0.0:8767.\n"),
        "glue row gone": ("docs/SUITE_GLUE.md", "| hook | 127.0.0.1:8770 |\n"),
        "overlay drops i2sd": ("docs/JUNIOROS_KERNEL.md", "overlay (os-release.junior)\n"),
    }
    for name, (rel, text) in bad_cases.items():
        docs = dict(_GOOD)
        docs[rel] = text
        cb, tb = run(docs, [].append)
        expect(cb["fail"] >= 1, f"bad case not caught: {name}")
        expect(tb != tip, f"bad case close unchanged: {name}")

    two = dict(_GOOD)
    two["docs/R.md"] = "Hook :8770  i2sd :8767  notes\n"
    ct, _ = run(two, [].append)
    expect(ct["fail"] == 0, f"label-then-port pair misread: {ct}")

    adv = dict(_GOOD)
    adv["docs/X.md"] = "UI :8765|:8771 and 127.0.0.1:8770/8771/8767\n"
    ca, _ = run(adv, [].append)
    expect(ca["fail"] == 0 and ca["advisory"] >= 2, f"advisory fixture: {ca}")

    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv) -> int:
    if "--selftest" in argv[1:]:
        return selftest()
    counts, _ = run(load_docs())
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
