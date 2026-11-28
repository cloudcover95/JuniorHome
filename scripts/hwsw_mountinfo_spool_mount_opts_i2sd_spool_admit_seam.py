#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/self/mountinfo lines -> i2sd spool admit plan.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the mount table line that covers the i2sd
  spool directory (flags, fstype, source) tells i2sd whether it may spool
  there. Hardware classes: SBC arm64 (JuniorOS T4, eMMC/SD, tighter) and SFF
  host (JuniorHome T0). Distinct from the block-queue, eMMC-wear, and
  block-mount GGUF seams (those read other files and other facts). Not T13
  (overnight owns it). Not item 2, 4, or 5.

What it emulates
  In-memory fixtures shaped like mountinfo text, for example
    36 25 259:2 / /var/lib/junior/spool rw,nosuid,nodev,noexec,noatime - ext4 /dev/nvme0n1p2 rw
  Rules, in order:
  1. Class must be "sbc" or "sff"; else hold (bad_class).
  2. The spool path must be absolute with no ".." part; else hold (bad_path).
  3. Every line must parse (>= 10 fields with a "-" separator); else hold
     (unparsed). The longest mount point that is a path-prefix of the spool
     path wins; none found -> hold (no_mount).
  4. fstype or source mentioning docker.sock/overlay-from-docker (source
     containing "docker.sock") -> reject (docker_sock). ROADMAP says no docker.sock.
  5. Mount option "ro" (either options field) -> hold (read_only).
  6. fstype "tmpfs" or "ramfs" -> hold (volatile): a spool must survive a reboot.
  7. Missing nosuid or nodev or noexec -> soft (opts_soft): admit with a
     smaller spool plan, since the spool should carry data only.
  8. fstype vfat/exfat/ntfs on an SBC -> soft (slow_fs).
  Plan: admit -> spool "full" (sbc 64 MiB, sff 256 MiB); soft -> half; hold or
  reject -> none. Each row packs decision and check states as 2-bit trit codes
  (0/1/2, four per byte, I2_S order: first code in the low bits); rows chain
  with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /proc or /sys reads, no sockets, no device opened, no subprocess,
  no mount calls, no network fetch, no model pull, no GGUF download. Rows go
  to a tempfile removed after the run; nothing under ~/.juniorhome. Flags
  mint, socket, layer_one, fetch, ue5_launch all False. No other team file or
  earlier hwsw_* seam is imported, read, or wrapped. JuniorSolana and the
  profile repo are untouched.

How to run
  python3 scripts/hwsw_mountinfo_spool_mount_opts_i2sd_spool_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_mountinfo_spool_mount_opts_i2sd_spool_admit_seam.py demo   # one line per fixture row
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
SPOOL_MIB = {"sbc": 64, "sff": 256}
VOLATILE = ("tmpfs", "ramfs")
SLOW_ON_SBC = ("vfat", "exfat", "ntfs", "ntfs3", "fuseblk")
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
RANK = {"admit": 0, "soft": 1, "hold": 2, "reject": 3}


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


def parse_mountinfo(text):
    """Return list of dicts (mp, opts, fstype, source) or None when unparsable."""
    rows = []
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    for ln in lines:
        f = ln.split()
        if len(f) < 10 or "-" not in f[6:]:
            return None
        s = f.index("-", 6)
        if len(f) < s + 4:
            return None
        rows.append({"mp": f[4], "opts": set(f[5].split(",")) | set(f[s + 3].split(",")),
                     "fstype": f[s + 1], "source": f[s + 2]})
    return rows


def covers(mp: str, path: str) -> bool:
    return mp == "/" or path == mp or path.startswith(mp.rstrip("/") + "/")


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    path = fx.get("spool", "")
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
    ok_path = path.startswith("/") and ".." not in path.split("/")
    mark("path", "admit" if ok_path else "hold", None if ok_path else "bad_path")
    rows = parse_mountinfo(fx.get("mountinfo"))
    mark("parse", "admit" if rows is not None else "hold", None if rows is not None else "unparsed")
    names = ("docker", "ro", "volatile", "opts", "slow_fs", "mount")
    hit = None
    if ok_cls and ok_path and rows is not None:
        cand = [r for r in rows if covers(r["mp"], path)]
        hit = max(cand, key=lambda r: len(r["mp"])) if cand else None
        mark("mount", "admit" if hit else "hold", None if hit else "no_mount")
    else:
        mark("mount", "admit")
    if hit is None:
        for k in names[:-1]:
            mark(k, "admit")
    else:
        dk = "docker.sock" in hit["source"] or "docker.sock" in hit["mp"]
        mark("docker", "reject" if dk else "admit", "docker_sock" if dk else None)
        ro = "ro" in hit["opts"]
        mark("ro", "hold" if ro else "admit", "read_only" if ro else None)
        vol = hit["fstype"] in VOLATILE
        mark("volatile", "hold" if vol else "admit", "volatile" if vol else None)
        miss = not {"nosuid", "nodev", "noexec"} <= hit["opts"]
        mark("opts", "soft" if miss else "admit", "opts_soft" if miss else None)
        slow = cls == "sbc" and hit["fstype"] in SLOW_ON_SBC
        mark("slow_fs", "soft" if slow else "admit", "slow_fs" if slow else None)
    mib = 0
    if worst == "admit" and ok_cls:
        mib = SPOOL_MIB[cls]
    elif worst == "soft" and ok_cls:
        mib = SPOOL_MIB[cls] // 2
    trits = [DECISION_CODE[worst]] + [DECISION_CODE[checks[k]] for k in
                                      ("cls", "path", "parse", "mount") + names[:-1]]
    return {"name": fx["name"], "cls": cls, "decision": worst, "admit": worst == "admit",
            "hold": flag, "plan": "spool" if mib else "none", "spool_mib": mib,
            "checks": checks, "trits": trits}


def mi(mp, opts, fstype, source, ropts="rw"):
    return f"36 25 259:2 / {mp} {opts} shared:1 - {fstype} {source} {ropts}"


ROOT = mi("/", "rw,relatime", "ext4", "/dev/mmcblk0p2")
SP = "/var/lib/junior/spool"
GOOD = mi(SP, "rw,nosuid,nodev,noexec,noatime", "ext4", "/dev/nvme0n1p2")

FIXTURES = [
    {"name": "sff_dedicated_spool_admit", "cls": "sff", "spool": SP, "mountinfo": ROOT + "\n" + GOOD,
     "want": {"decision": "admit", "plan": "spool", "spool_mib": 256, "hold": None}},
    {"name": "sbc_dedicated_spool_admit", "cls": "sbc", "spool": SP, "mountinfo": ROOT + "\n" + GOOD,
     "want": {"decision": "admit", "spool_mib": 64}},
    {"name": "longest_mount_wins_over_root", "cls": "sff", "spool": SP + "/q1", "mountinfo": GOOD + "\n" + ROOT,
     "want": {"decision": "admit", "spool_mib": 256}},
    {"name": "root_only_missing_opts_soft", "cls": "sff", "spool": SP, "mountinfo": ROOT,
     "want": {"decision": "soft", "hold": "opts_soft", "spool_mib": 128}},
    {"name": "sbc_root_only_soft_half", "cls": "sbc", "spool": SP, "mountinfo": ROOT,
     "want": {"decision": "soft", "spool_mib": 32}},
    {"name": "sibling_prefix_not_covered", "cls": "sff", "spool": SP + "-old", "mountinfo": ROOT + "\n" + GOOD,
     "want": {"decision": "soft", "hold": "opts_soft"}},
    {"name": "read_only_hold", "cls": "sff", "spool": SP,
     "mountinfo": mi(SP, "ro,nosuid,nodev,noexec", "ext4", "/dev/sda1", "ro"),
     "want": {"decision": "hold", "hold": "read_only", "plan": "none"}},
    {"name": "super_option_ro_hold", "cls": "sbc", "spool": SP,
     "mountinfo": mi(SP, "rw,nosuid,nodev,noexec", "ext4", "/dev/sda1", "ro,errors=remount-ro"),
     "want": {"decision": "hold", "hold": "read_only"}},
    {"name": "tmpfs_volatile_hold", "cls": "sff", "spool": SP,
     "mountinfo": mi(SP, "rw,nosuid,nodev,noexec", "tmpfs", "tmpfs"),
     "want": {"decision": "hold", "hold": "volatile", "spool_mib": 0}},
    {"name": "docker_sock_source_reject", "cls": "sff", "spool": SP,
     "mountinfo": mi(SP, "rw,nosuid,nodev,noexec", "ext4", "/run/docker.sock"),
     "want": {"decision": "reject", "hold": "docker_sock", "plan": "none"}},
    {"name": "sbc_vfat_soft_slow_fs", "cls": "sbc", "spool": SP,
     "mountinfo": mi(SP, "rw,nosuid,nodev,noexec", "vfat", "/dev/mmcblk0p1"),
     "want": {"decision": "soft", "hold": "slow_fs", "spool_mib": 32}},
    {"name": "sff_vfat_ok", "cls": "sff", "spool": SP,
     "mountinfo": mi(SP, "rw,nosuid,nodev,noexec", "vfat", "/dev/sdb1"),
     "want": {"decision": "admit"}},
    {"name": "no_covering_mount_hold", "cls": "sff", "spool": SP, "mountinfo": mi("/mnt/usb", "rw", "ext4", "/dev/sdb1"),
     "want": {"decision": "hold", "hold": "no_mount"}},
    {"name": "garbage_hold", "cls": "sbc", "spool": SP, "mountinfo": "garbage\n",
     "want": {"decision": "hold", "hold": "unparsed", "plan": "none"}},
    {"name": "empty_hold", "cls": "sff", "spool": SP, "mountinfo": "",
     "want": {"decision": "hold", "hold": "unparsed"}},
    {"name": "dotdot_path_hold", "cls": "sff", "spool": "/var/../etc", "mountinfo": ROOT,
     "want": {"decision": "hold", "hold": "bad_path"}},
    {"name": "relative_path_hold", "cls": "sbc", "spool": "spool", "mountinfo": ROOT,
     "want": {"decision": "hold", "hold": "bad_path"}},
    {"name": "bad_class_hold", "cls": "laptop", "spool": SP, "mountinfo": ROOT + "\n" + GOOD,
     "want": {"decision": "hold", "hold": "bad_class", "plan": "none"}},
]


def run(verbose: bool = False) -> dict:
    prev, rows, fails = GENESIS, [], []
    fd, path = tempfile.mkstemp(prefix="hwsw_mountinfo_", suffix=".jsonl")
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
                          f"spool_mib={r['spool_mib']} close={prev}")
    finally:
        os.remove(path)
    pack_ok = pack2([2, 0, 1, 1]) == bytes([0b01010010])
    return {"seam": "mountinfo_spool->i2sd_spool_admit",
            "roadmap": "ROADMAP #3 i2sd userspace 127.0.0.1:8767 no docker.sock",
            "rate": RATE, "flags": FLAGS, "rows": len(rows), "pack_order_ok": pack_ok, "fail": fails,
            "pass": not fails and pack_ok and not any(FLAGS.values()), "tip_close": prev}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")} if demo else out, indent=2))
    sys.exit(0 if out["pass"] else 1)
