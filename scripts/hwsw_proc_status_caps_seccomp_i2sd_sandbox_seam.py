#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/<pid>/status CapEff + NoNewPrivs + Seccomp -> i2sd sandbox admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the kernel-reported privilege state of the
  i2sd process (effective capability mask, no_new_privs, seccomp mode) decides
  whether it may start. Hardware classes: SBC arm64 (JuniorOS T4) and SFF host
  (JuniorHome T0). Not T13, not item 2, not item 4, not item 5. No other team
  file or earlier hwsw_* seam is imported, read, or wrapped; process status
  capabilities were not covered before.

What it emulates
  In-memory fixtures shaped like the "CapEff:", "NoNewPrivs:" and "Seccomp:"
  lines of /proc/<pid>/status (hex mask string, "0"/"1", "0"/"1"/"2").
  Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. Any field unparsed (non-hex mask, out-of-range mode) -> hold (unparsed).
  4. CapEff has CAP_SYS_ADMIN (21), CAP_NET_ADMIN (12), CAP_SYS_PTRACE (19),
     CAP_SYS_MODULE (16) or CAP_DAC_READ_SEARCH (2) -> reject (cap_too_wide).
  5. Seccomp 1 (strict) -> hold (seccomp_strict): i2sd needs read/write/accept.
  6. NoNewPrivs 0 -> soft (no_new_privs_off), threads halved (min 1).
  7. Seccomp 0 -> soft (seccomp_off), threads halved (min 1).
  8. Otherwise admit (CapEff 0 or only low-risk bits, NoNewPrivs 1, Seccomp 2).
  Threads: SBC 4, SFF 8, capped by wanted; hold/reject gives 0. Rows pack
  decision and check states as 2-bit trit codes (four per byte, I2_S order,
  first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no prctl, no seccomp install, no capset, no
  systemd, no sockets (binds are strings), no subprocess, no network fetch,
  no model pull, no GGUF download. Rows live in memory only. Flags mint,
  socket, layer_one, fetch, ue5_launch all False. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_proc_status_caps_seccomp_i2sd_sandbox_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_proc_status_caps_seccomp_i2sd_sandbox_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
CLASS_THREADS = {"sbc": 4, "sff": 8}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
BAD_CAPS = {2: "dac_read_search", 12: "net_admin", 16: "sys_module", 19: "sys_ptrace", 21: "sys_admin"}


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


def parse_caps(s):
    s = str(s).strip()
    if not s or len(s) > 16 or any(ch not in "0123456789abcdefABCDEF" for ch in s):
        return None
    return int(s, 16)


def parse_mode(s, hi):
    s = str(s).strip()
    return int(s) if s.isdigit() and int(s) <= hi else None


def decide(fx):
    """Return (decision, reason, threads, codes)."""
    def done(d, r, codes):
        t = 0
        if d in ("admit", "soft"):
            t = min(CLASS_THREADS[fx["cls"]], fx.get("want", 99))
            if d == "soft":
                t = max(1, t // 2)
        return d, r, t, [DECISION_CODE[d]] + codes

    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASS_THREADS:
        return "hold", "bad_class", 0, [2, 2, 0]
    caps = parse_caps(fx.get("cap_eff"))
    nnp = parse_mode(fx.get("no_new_privs"), 1)
    sec = parse_mode(fx.get("seccomp"), 2)
    if caps is None or nnp is None or sec is None:
        return done("hold", "unparsed", [2, 2, 0])
    for bit in sorted(BAD_CAPS):
        if caps >> bit & 1:
            return done("reject", "cap_too_wide", [2, 2, 2, 0])
    if sec == 1:
        return done("hold", "seccomp_strict", [2, 2, 2, 2, 0])
    if nnp == 0:
        return done("soft", "no_new_privs_off", [2, 2, 2, 2, 1])
    if sec == 0:
        return done("soft", "seccomp_off", [2, 2, 2, 2, 2, 1])
    return done("admit", "ok", [2, 2, 2, 2, 2, 2])


FIXTURES = [
    ("sbc_locked", dict(cls="sbc", bind=BIND_OK, want=8, cap_eff="0000000000000000", no_new_privs="1", seccomp="2"), "admit", 4),
    ("sff_locked", dict(cls="sff", bind=BIND_OK, want=8, cap_eff="0000000000000000", no_new_privs="1", seccomp="2"), "admit", 8),
    ("sff_net_bind_only", dict(cls="sff", bind=BIND_OK, want=8, cap_eff="0000000000000400", no_new_privs="1", seccomp="2"), "admit", 8),
    ("sff_nnp_off", dict(cls="sff", bind=BIND_OK, want=8, cap_eff="0", no_new_privs="0", seccomp="2"), "soft", 4),
    ("sbc_seccomp_off", dict(cls="sbc", bind=BIND_OK, want=8, cap_eff="0", no_new_privs="1", seccomp="0"), "soft", 2),
    ("sbc_nnp_off_want1", dict(cls="sbc", bind=BIND_OK, want=1, cap_eff="0", no_new_privs="0", seccomp="2"), "soft", 1),
    ("sff_strict", dict(cls="sff", bind=BIND_OK, want=8, cap_eff="0", no_new_privs="1", seccomp="1"), "hold", 0),
    ("sff_sys_admin", dict(cls="sff", bind=BIND_OK, want=8, cap_eff="0000000000200000", no_new_privs="1", seccomp="2"), "reject", 0),
    ("sbc_net_admin", dict(cls="sbc", bind=BIND_OK, want=8, cap_eff="1000", no_new_privs="1", seccomp="2"), "reject", 0),
    ("sff_full_caps", dict(cls="sff", bind=BIND_OK, want=8, cap_eff="000001ffffffffff", no_new_privs="0", seccomp="0"), "reject", 0),
    ("unparsed_mask", dict(cls="sff", bind=BIND_OK, cap_eff="zz", no_new_privs="1", seccomp="2"), "hold", 0),
    ("seccomp_out_of_range", dict(cls="sff", bind=BIND_OK, cap_eff="0", no_new_privs="1", seccomp="3"), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, cap_eff="0", no_new_privs="1", seccomp="2"), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", cap_eff="0", no_new_privs="1", seccomp="2"), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, cap_eff="0", no_new_privs="1", seccomp="2"), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_t in FIXTURES:
        d, reason, t, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, d, reason, t))
        rows.append({"name": name, "decision": d, "reason": reason, "threads": t, "pack": pk.hex(), "close": prev})
        if (d, t) != (want_d, want_t):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-21s %-6s %-16s t=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["threads"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
