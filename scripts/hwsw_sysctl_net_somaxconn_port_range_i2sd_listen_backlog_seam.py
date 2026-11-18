#!/usr/bin/env python3
"""Hardware-to-software seam emulator: net sysctl (somaxconn, tcp_max_syn_backlog, ip_local_port_range) -> i2sd listen backlog.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where kernel network sysctls decide the listen()
  backlog i2sd may request on its loopback socket and whether the ephemeral
  port range collides with the fixed i2sd port. Hardware classes: SBC arm64
  (JuniorOS T4) and SFF host (JuniorHome T0). Not T13, not item 2 (no GGUF),
  not item 4 (no Asahi/MLX), not item 5. No other team file or earlier hwsw_*
  seam is imported, read, or wrapped; no net sysctl file was covered before.

What it emulates
  In-memory fixtures shaped like /proc/sys/net/core/somaxconn,
  /proc/sys/net/ipv4/tcp_max_syn_backlog and /proc/sys/net/ipv4/ip_local_port_range
  as text the kernel would return. Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Any value that does not parse as the expected integers -> hold (unparsed).
  4. Port range containing 8767 -> soft (port_in_ephemeral): i2sd could lose
     its port to an outgoing connection; backlog is halved.
  5. somaxconn < 16 -> soft (tiny_somaxconn); backlog = somaxconn.
  6. Otherwise admit; backlog = min(wanted, somaxconn, tcp_max_syn_backlog,
     class cap) with class cap SBC 64, SFF 128.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sysctl writes, no sysctl binary, no
  sockets (binds are strings), no listen() call, no device opened, no network
  fetch, no subprocess, no model pull, no GGUF download. Rows live in memory
  only. Flags mint, socket, layer_one, fetch, ue5_launch all False.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_sysctl_net_somaxconn_port_range_i2sd_listen_backlog_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_sysctl_net_somaxconn_port_range_i2sd_listen_backlog_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
PORT = 8767
CLASS_CAP = {"sbc": 64, "sff": 128}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes):
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


def parse(fx):
    """Parse sysctl text; return (somaxconn, syn, lo, hi) or None."""
    try:
        so = int(fx["somaxconn"].strip())
        syn = int(fx["syn_backlog"].strip())
        parts = fx["port_range"].split()
        if len(parts) != 2:
            return None
        lo, hi = int(parts[0]), int(parts[1])
    except (KeyError, ValueError, AttributeError):
        return None
    if so < 1 or syn < 1 or not (0 < lo <= hi <= 65535):
        return None
    return so, syn, lo, hi


def decide(fx):
    """Return (decision, reason, backlog, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASS_CAP:
        return "hold", "bad_class", 0, [2, 2, 0]
    p = parse(fx)
    if p is None:
        return "hold", "unparsed", 0, [2, 2, 0]
    so, syn, lo, hi = p
    cap = min(fx.get("want", 999), so, syn, CLASS_CAP[fx["cls"]])
    if lo <= PORT <= hi:
        return "soft", "port_in_ephemeral", max(1, cap // 2), [1, 2, 1]
    if so < 16:
        return "soft", "tiny_somaxconn", so, [1, 1, 2]
    return "admit", "ok", cap, [2, 2, 2]


def fx(cls, so, syn, pr, want=999, bind=BIND_OK, **kw):
    d = dict(cls=cls, somaxconn=so, syn_backlog=syn, port_range=pr, want=want, bind=bind)
    d.update(kw)
    return d


FIXTURES = [
    ("sff_default", fx("sff", "4096\n", "1024\n", "32768\t60999\n"), "admit", 128),
    ("sbc_default", fx("sbc", "4096\n", "1024\n", "32768\t60999\n"), "admit", 64),
    ("sbc_old_somaxconn", fx("sbc", "128\n", "128\n", "32768\t60999\n"), "admit", 64),
    ("sff_want_small", fx("sff", "4096\n", "1024\n", "32768\t60999\n", want=10), "admit", 10),
    ("sff_syn_limits", fx("sff", "4096\n", "32\n", "32768\t60999\n"), "admit", 32),
    ("sbc_tiny_somaxconn", fx("sbc", "8\n", "128\n", "32768\t60999\n"), "soft", 8),
    ("sff_port_in_range", fx("sff", "4096\n", "1024\n", "1024\t65535\n"), "soft", 64),
    ("sbc_port_in_range_min", fx("sbc", "1\n", "1\n", "8000\t9000\n"), "soft", 1),
    ("bad_class", fx("x86mac", "4096", "1024", "32768 60999"), "hold", 0),
    ("unparsed_range", fx("sff", "4096", "1024", "32768"), "hold", 0),
    ("unparsed_somax", fx("sff", "abc", "1024", "32768 60999"), "hold", 0),
    ("bind_any", fx("sff", "4096", "1024", "32768 60999", bind="0.0.0.0:8767"), "reject", 0),
    ("docker_sock", fx("sff", "4096", "1024", "32768 60999", docker_sock=True), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, f_, want_d, want_b in FIXTURES:
        dec, reason, b, codes = decide(f_)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, dec, reason, b))
        rows.append({"name": name, "decision": dec, "reason": reason, "backlog": b, "pack": pk.hex(), "close": prev})
        if (dec, b) != (want_d, want_b):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-22s %-6s %-18s b=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["backlog"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
