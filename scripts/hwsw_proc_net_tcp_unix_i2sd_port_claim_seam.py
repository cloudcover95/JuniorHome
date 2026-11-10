#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/net/tcp + /proc/net/unix listener tables -> i2sd port-claim plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the kernel's socket tables (text rows of
  /proc/net/tcp and /proc/net/unix) tell i2sd whether it may claim
  127.0.0.1:8767 at start, is already up, or must refuse because a wildcard
  listener or a docker.sock listener exists. Hardware classes: SBC arm64
  (<=12 W, JuniorOS T4) and SFF host (JuniorHome T0); the tables are
  little-endian hex on both. Distinct from the net-class sysfs bind-health
  seam (no link/carrier/operstate), the tun/loopback overlay seam (no
  interface state), and the hwmon/thermal/power/memory/cgroup seams. Not T13
  (overnight owns it). Not item 2, 4, or 5. No other team file or earlier
  hwsw_* seam is imported, read, or wrapped.

What it emulates
  In-memory fixtures shaped like the kernel's text: the /proc/net/tcp header
  and rows (sl local_address rem_address st ...; address is hex IPv4 as
  little-endian bytes, so 127.0.0.1 is 0100007F; port is big-endian hex, so
  8767 is 223F and 8765 is 223D; st 0A is LISTEN, 01 is ESTABLISHED) and a
  /proc/net/unix listing (Flags 00010000 marks a listening socket). Rules,
  in order:
  1. Class must be "sbc" or "sff"; else hold (bad_class).
  2. tcp table must have the "local_address" header and every row must parse
     (hex addr:port, 2-hex state); else hold (unparsed). Same for unix.
  3. Any unix listener whose path ends in docker.sock -> reject (docker_sock).
  4. A LISTEN row on 8767 whose address is not 0100007F (0.0.0.0, a LAN
     address, ...) -> reject (wildcard_listen). A LISTEN row on 8765
     (bitnetd) not on 0100007F -> reject (bitnetd_wildcard).
  5. A loopback LISTEN row on 8767: if the fixture says i2sd_running True it
     is ours -> plan "already_up"; else another process holds it -> hold
     (port_taken). No such row -> plan "start".
  6. bitnetd loopback LISTEN on 8765 absent -> soft (bitnetd_absent): i2sd
     may start, but its upstream is not there.
  7. ESTABLISHED rows with local port 8767 above 64 -> soft (conns_high).
  Decision is the worst row. Each row packs decision and check states as
  2-bit trit codes (0/1/2, four per byte, I2_S order: first code in the low
  bits); rows chain with close rule sha256(prev || pack || text)[:16] at
  rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sockets (nothing listens, binds, or
  connects; ports are parsed from strings), no netlink, no ss/netstat
  subprocess, no systemd calls, no device opened, no network fetch, no
  model pull, no GGUF download. Rows go to a tempfile removed after the run;
  nothing under ~/.juniorhome. Flags mint, socket, layer_one, fetch,
  ue5_launch all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_proc_net_tcp_unix_i2sd_port_claim_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_proc_net_tcp_unix_i2sd_port_claim_seam.py demo   # one line per fixture row
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
I2SD_PORT, BITNETD_PORT = 8767, 8765
LOOPBACK = "0100007F"
CONN_CAP = 64
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
RANK = {"admit": 0, "soft": 1, "hold": 2, "reject": 3}

TCP_HEAD = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
UNIX_HEAD = "Num       RefCount Protocol Flags    Type St Inode Path\n"


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


TCP_ROW = re.compile(r"^\s*\d+:\s+([0-9A-Fa-f]{8}):([0-9A-Fa-f]{4})\s+[0-9A-Fa-f]{8}:[0-9A-Fa-f]{4}\s+([0-9A-Fa-f]{2})\b")


def parse_tcp(text):
    """Return list of (addr_hex, port, state_hex) or None when unparsable."""
    lines = (text or "").splitlines()
    if not lines or "local_address" not in lines[0]:
        return None
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        m = TCP_ROW.match(ln)
        if not m:
            return None
        rows.append((m.group(1).upper(), int(m.group(2), 16), m.group(3).upper()))
    return rows


def parse_unix(text):
    """Return list of (flags_int, path) or None when unparsable."""
    lines = (text or "").splitlines()
    if not lines or "Path" not in lines[0]:
        return None
    rows = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 7 or not re.fullmatch(r"[0-9A-Fa-f]{8}", parts[3]):
            return None
        rows.append((int(parts[3], 16), parts[7] if len(parts) > 7 else ""))
    return rows


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

    mark("cls", "admit" if cls in CLASSES else "hold", None if cls in CLASSES else "bad_class")
    tcp, uni = parse_tcp(fx.get("tcp")), parse_unix(fx.get("unix"))
    mark("parse", "admit" if tcp is not None and uni is not None else "hold",
         None if tcp is not None and uni is not None else "unparsed")
    plan = "none"
    if tcp is None or uni is None:
        mark("docker", "admit"); mark("wild", "admit"); mark("claim", "admit")
        mark("upstream", "admit"); mark("conns", "admit")
    else:
        dock = any(fl & 0x10000 and p.endswith("docker.sock") for fl, p in uni)
        mark("docker", "reject" if dock else "admit", "docker_sock" if dock else None)
        listen = [(a, pt) for a, pt, st in tcp if st == "0A"]
        w_i2sd = any(pt == I2SD_PORT and a != LOOPBACK for a, pt in listen)
        w_bit = any(pt == BITNETD_PORT and a != LOOPBACK for a, pt in listen)
        if w_i2sd:
            mark("wild", "reject", "wildcard_listen")
        elif w_bit:
            mark("wild", "reject", "bitnetd_wildcard")
        else:
            mark("wild", "admit")
        held = any(pt == I2SD_PORT and a == LOOPBACK for a, pt in listen)
        if held and not fx.get("i2sd_running"):
            mark("claim", "hold", "port_taken")
        else:
            mark("claim", "admit")
            plan = "already_up" if held else "start"
        up = any(pt == BITNETD_PORT and a == LOOPBACK for a, pt in listen)
        mark("upstream", "admit" if up else "soft", None if up else "bitnetd_absent")
        est = sum(1 for a, pt, st in tcp if pt == I2SD_PORT and st == "01")
        mark("conns", "soft" if est > CONN_CAP else "admit", "conns_high" if est > CONN_CAP else None)
    if worst in ("hold", "reject"):
        plan = "none"
    trits = [DECISION_CODE[worst]] + [DECISION_CODE[checks[k]] for k in
                                      ("cls", "parse", "docker", "wild", "claim", "upstream", "conns")]
    return {"name": fx["name"], "cls": cls, "decision": worst, "admit": worst == "admit",
            "hold": flag, "plan": plan, "checks": checks, "trits": trits}


def trow(sl, addr, port, st):
    return "%4d: %s:%04X 00000000:0000 %s 00000000:00000000 00:00000000 00000000     0        0 %d 1 0\n" % (
        sl, addr, port, st, 1000 + sl)


def tcp(*rows):
    return TCP_HEAD + "".join(trow(i, *r) for i, r in enumerate(rows))


def urow(i, flags, path):
    return "%016x: 00000002 00000000 %08X 0001 %02d %d %s\n" % (i, flags, 1 if flags & 0x10000 else 3, 2000 + i, path)


UNIX_QUIET = UNIX_HEAD + urow(1, 0x10000, "/run/systemd/private") + urow(2, 0, "/run/dbus/system_bus_socket")
UNIX_DOCKER = UNIX_QUIET + urow(3, 0x10000, "/var/run/docker.sock")
UNIX_DOCKER_CLIENT = UNIX_QUIET + urow(3, 0, "/var/run/docker.sock")
BIT = (LOOPBACK, 8765, "0A")
I2SD = (LOOPBACK, 8767, "0A")

FIXTURES = [
    {"name": "sff_free_start", "cls": "sff", "tcp": tcp(BIT), "unix": UNIX_QUIET,
     "want": {"decision": "admit", "plan": "start", "hold": None}},
    {"name": "sbc_free_start", "cls": "sbc", "tcp": tcp(BIT, (LOOPBACK, 22, "0A")), "unix": UNIX_QUIET,
     "want": {"decision": "admit", "plan": "start"}},
    {"name": "sff_already_up", "cls": "sff", "i2sd_running": True, "tcp": tcp(BIT, I2SD),
     "unix": UNIX_QUIET, "want": {"decision": "admit", "plan": "already_up"}},
    {"name": "sbc_port_taken_hold", "cls": "sbc", "tcp": tcp(BIT, I2SD), "unix": UNIX_QUIET,
     "want": {"decision": "hold", "hold": "port_taken", "plan": "none"}},
    {"name": "sff_wildcard_i2sd_reject", "cls": "sff", "tcp": tcp(BIT, ("00000000", 8767, "0A")),
     "unix": UNIX_QUIET, "want": {"decision": "reject", "hold": "wildcard_listen", "admit": False}},
    {"name": "sbc_lan_addr_i2sd_reject", "cls": "sbc", "tcp": tcp(BIT, ("0100A8C0", 8767, "0A")),
     "unix": UNIX_QUIET, "want": {"decision": "reject", "hold": "wildcard_listen"}},
    {"name": "sff_wildcard_bitnetd_reject", "cls": "sff", "tcp": tcp(("00000000", 8765, "0A")),
     "unix": UNIX_QUIET, "want": {"decision": "reject", "hold": "bitnetd_wildcard"}},
    {"name": "sff_docker_sock_listen_reject", "cls": "sff", "tcp": tcp(BIT), "unix": UNIX_DOCKER,
     "want": {"decision": "reject", "hold": "docker_sock", "plan": "none"}},
    {"name": "sbc_docker_sock_client_ok", "cls": "sbc", "tcp": tcp(BIT), "unix": UNIX_DOCKER_CLIENT,
     "want": {"decision": "admit", "plan": "start"}},
    {"name": "sbc_bitnetd_absent_soft", "cls": "sbc", "tcp": tcp((LOOPBACK, 22, "0A")), "unix": UNIX_QUIET,
     "want": {"decision": "soft", "hold": "bitnetd_absent", "plan": "start"}},
    {"name": "sff_conns_high_soft", "cls": "sff", "i2sd_running": True,
     "tcp": tcp(BIT, I2SD, *[(LOOPBACK, 8767, "01")] * 65), "unix": UNIX_QUIET,
     "want": {"decision": "soft", "hold": "conns_high", "plan": "already_up"}},
    {"name": "sff_conns_at_cap_ok", "cls": "sff", "i2sd_running": True,
     "tcp": tcp(BIT, I2SD, *[(LOOPBACK, 8767, "01")] * 64), "unix": UNIX_QUIET,
     "want": {"decision": "admit", "plan": "already_up"}},
    {"name": "sbc_other_port_wildcard_ok", "cls": "sbc", "tcp": tcp(BIT, ("00000000", 8080, "0A")),
     "unix": UNIX_QUIET, "want": {"decision": "admit", "plan": "start"}},
    {"name": "sff_missing_tcp_header_hold", "cls": "sff", "tcp": "garbage\n", "unix": UNIX_QUIET,
     "want": {"decision": "hold", "hold": "unparsed", "plan": "none"}},
    {"name": "sbc_bad_tcp_row_hold", "cls": "sbc", "tcp": TCP_HEAD + "   0: ZZZZ:223F x y\n",
     "unix": UNIX_QUIET, "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "sff_missing_unix_header_hold", "cls": "sff", "tcp": tcp(BIT), "unix": "",
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "bad_class_hold", "cls": "laptop", "tcp": tcp(BIT), "unix": UNIX_QUIET,
     "want": {"decision": "hold", "hold": "bad_class", "plan": "none"}},
]


def run(verbose: bool = False) -> dict:
    prev, rows, fails = GENESIS, [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_portclaim_", suffix=".jsonl")
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
                    print(f"{r['name']:<32} decision={r['decision']:<7} hold={r['hold']!s:<16} "
                          f"plan={r['plan']:<10} close={prev}")
    finally:
        os.remove(path)
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    port_ok = (int("223F", 16), int("223D", 16)) == (I2SD_PORT, BITNETD_PORT)
    return {"seam": "proc_net_tcp_unix->i2sd_port_claim",
            "roadmap": "ROADMAP #3 i2sd userspace 127.0.0.1:8767 no docker.sock",
            "rate": RATE, "flags": FLAGS, "rows": len(rows),
            "pack_order_ok": pack_ok, "port_hex_ok": port_ok, "fail": fails,
            "pass": not fails and pack_ok and port_ok and not any(FLAGS.values()), "tip_close": prev}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    if not demo:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")}, indent=2))
    sys.exit(0 if out["pass"] else 1)
