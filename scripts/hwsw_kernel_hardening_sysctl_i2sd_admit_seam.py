#!/usr/bin/env python3
"""Hardware-to-software seam emulator: kernel hardening sysctls -> i2sd admit plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the host's kernel hardening knobs
  (kernel.yama.ptrace_scope, kernel.kptr_restrict, kernel.dmesg_restrict,
  kernel.unprivileged_bpf_disabled, kernel.perf_event_paranoid,
  user.max_user_namespaces) decide whether i2sd may start as an unprivileged
  userspace service. Hardware classes: SBC arm64 (JuniorOS T4) and SFF host
  (JuniorHome T0). Distinct from the caps/seccomp (/proc status), net sysctl,
  vm sysctl, and cgroup seams: those read other files. Not T13 (overnight
  owns it). Not item 2, 4, or 5.

What it emulates
  In-memory fixtures shaped like the text of /proc/sys files (one value each).
  Rules, in order:
  1. Class must be "sbc" or "sff"; else hold (bad_class).
  2. Every knob must parse as an int; else hold (unparsed).
  3. unprivileged_bpf_disabled == 0 -> reject (bpf_open): unprivileged eBPF is
     open, a loopback service must not start beside it.
  4. ptrace_scope == 0 -> reject (ptrace_open): any same-uid process could
     read i2sd memory.
  5. kptr_restrict == 0 or dmesg_restrict == 0 -> hold (leaky_kernel).
  6. perf_event_paranoid below 2 (sbc) or below 1 (sff) -> hold (perf_open).
  7. max_user_namespaces above 0 -> soft (userns_on): start, but with the
     reduced plan (half workers, no helper process).
  Plan: admit -> "start" with workers 4 (sbc) or 8 (sff); soft -> half;
  hold or reject -> none. Each row packs decision and check states as 2-bit
  trit codes (0/1/2, four per byte, I2_S order: first code in the low bits);
  rows chain with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sockets, no device opened, no subprocess, no
  systemd calls, no network fetch, no model pull, no GGUF download. Rows go to
  a tempfile removed after the run; nothing under ~/.juniorhome. Flags mint,
  socket, layer_one, fetch, ue5_launch all False. No other team file or earlier
  hwsw_* seam is imported, read, or wrapped. JuniorSolana and the profile repo
  are untouched.

How to run
  python3 scripts/hwsw_kernel_hardening_sysctl_i2sd_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_kernel_hardening_sysctl_i2sd_admit_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
GENESIS = "0" * 16
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
WORKERS = {"sbc": 4, "sff": 8}
PERF_MIN = {"sbc": 2, "sff": 1}
KNOBS = ("ptrace_scope", "kptr_restrict", "dmesg_restrict", "unpriv_bpf_disabled",
         "perf_event_paranoid", "max_user_namespaces")


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes: list) -> bytes:
    out = bytearray((len(codes) + 3) // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= (c & 3) << (2 * (i % 4))
    return bytes(out)


def parse_int(text):
    try:
        return int(str(text).strip())
    except (ValueError, TypeError):
        return None


def evaluate(fx: dict) -> dict:
    cls = fx.get("class")
    marks = {}

    def mark(name, state):
        marks[name] = state  # 0 pass, 1 soft/hold, 2 reject

    if cls not in CLASSES:
        return {"decision": "hold", "reason": "bad_class", "workers": 0, "trits": [1, 0, 0, 0, 0, 0]}
    v = {k: parse_int(fx.get(k)) for k in KNOBS}
    if any(x is None for x in v.values()):
        return {"decision": "hold", "reason": "unparsed", "workers": 0, "trits": [0, 1, 0, 0, 0, 0]}
    mark("bpf", 2 if v["unpriv_bpf_disabled"] == 0 else 0)
    mark("ptrace", 2 if v["ptrace_scope"] == 0 else 0)
    mark("leak", 1 if (v["kptr_restrict"] == 0 or v["dmesg_restrict"] == 0) else 0)
    mark("perf", 1 if v["perf_event_paranoid"] < PERF_MIN[cls] else 0)
    mark("userns", 1 if v["max_user_namespaces"] > 0 else 0)
    if marks["bpf"] == 2:
        d, r = "reject", "bpf_open"
    elif marks["ptrace"] == 2:
        d, r = "reject", "ptrace_open"
    elif marks["leak"]:
        d, r = "hold", "leaky_kernel"
    elif marks["perf"]:
        d, r = "hold", "perf_open"
    elif marks["userns"]:
        d, r = "soft", "userns_on"
    else:
        d, r = "admit", "ok"
    w = {"admit": WORKERS[cls], "soft": WORKERS[cls] // 2}.get(d, 0)
    dcode = {"admit": 0, "soft": 1, "hold": 1, "reject": 2}[d]
    trits = [dcode, 0, marks["bpf"], marks["ptrace"], marks["leak"], marks["perf"], marks["userns"]]
    return {"decision": d, "reason": r, "workers": w, "trits": trits}


def fx(cls, ptrace=1, kptr=1, dmesg=1, bpf=2, perf=2, userns=0):
    return {"class": cls, "ptrace_scope": "%s\n" % ptrace, "kptr_restrict": "%s\n" % kptr,
            "dmesg_restrict": "%s\n" % dmesg, "unpriv_bpf_disabled": "%s\n" % bpf,
            "perf_event_paranoid": "%s\n" % perf, "max_user_namespaces": "%s\n" % userns}


ROWS = [
    ("sbc clean", fx("sbc"), "admit", 4),
    ("sff clean", fx("sff"), "admit", 8),
    ("bad class", fx("x86"), "hold", 0),
    ("unparsed", dict(fx("sbc"), ptrace_scope="n/a"), "hold", 0),
    ("bpf open sbc", fx("sbc", bpf=0), "reject", 0),
    ("bpf open sff", fx("sff", bpf=0), "reject", 0),
    ("bpf locked once is ok", fx("sbc", bpf=1), "admit", 4),
    ("ptrace open", fx("sff", ptrace=0), "reject", 0),
    ("kptr leaky", fx("sbc", kptr=0), "hold", 0),
    ("dmesg leaky", fx("sff", dmesg=0), "hold", 0),
    ("perf 1 on sbc", fx("sbc", perf=1), "hold", 0),
    ("perf 1 on sff ok", fx("sff", perf=1), "admit", 8),
    ("perf 0 on sff", fx("sff", perf=0), "hold", 0),
    ("userns on sbc", fx("sbc", userns=31000), "soft", 2),
    ("userns on sff", fx("sff", userns=62000), "soft", 4),
    ("bpf beats userns", fx("sff", bpf=0, userns=5), "reject", 0),
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    fails = []
    path = None
    try:
        fd, path = tempfile.mkstemp(prefix="hwsw_khard_")
        with os.fdopen(fd, "w") as fh:
            for name, f, want, wk in ROWS:
                r = evaluate(f)
                text = json.dumps({"n": name, "d": r["decision"], "r": r["reason"], "w": r["workers"]}, sort_keys=True)
                prev = close(prev, pack2(r["trits"]), text)
                fh.write(text + "\n")
                if verbose:
                    print("%-24s %-7s %-13s workers=%d close=%s" % (name, r["decision"], r["reason"], r["workers"], prev))
                if r["decision"] != want or r["workers"] != wk:
                    fails.append(name)
    finally:
        if path and os.path.exists(path):
            os.remove(path)
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    return {"seam": "kernel_hardening_sysctl_i2sd_admit", "roadmap": "ROADMAP.md item 3 i2sd",
            "rate": RATE, "flags": FLAGS, "rows": len(ROWS), "pack_order_ok": pack_ok, "fail": fails,
            "pass": not fails and pack_ok and not any(FLAGS.values()), "tip_close": prev}


def main(argv: list) -> int:
    res = run(verbose=len(argv) > 1 and argv[1] == "demo")
    print(json.dumps(res, sort_keys=True))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
