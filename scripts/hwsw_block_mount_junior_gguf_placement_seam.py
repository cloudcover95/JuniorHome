#!/usr/bin/env python3
"""Hardware-to-software seam emulator: block device + mount table -> where JUNIOR_GGUF may live.

Roadmap item
  JuniorHome docs/ROADMAP.md item 2 (line 18), "GGUF — operator sets
  JUNIOR_GGUF. Bots do not download.", at the storage seam: the operator has
  pointed JUNIOR_GGUF at a path, and bitnetd.service (127.0.0.1:8765) has to
  decide whether the filesystem and block device under that path are a safe
  place to map weights from on its hardware class (SBC arm64 <=12 W ->
  JuniorOS T4, SFF host x86_64/arm64 -> JuniorHome T0). Rule: storage only
  admits, holds or rejects; bitnetd never copies, moves, re-downloads or
  caches the model somewhere else on its own. A hold always means "operator
  action needed". Not ROADMAP T13 (overnight owns it); no juniorctl change.
  Not a re-derivation of any earlier seam file: no other team file is
  imported or read (the memory/KV seam answered "does it fit in RAM"; this
  one answers "is the storage under it local, fixed and readable").

What it emulates
  In-memory fixtures shaped like kernel text:
    /proc/mounts lines              ("<src> <mnt> <fstype> <opts> 0 0")
    /sys/block/<dev>/removable      ("0" / "1")
    /sys/block/<dev>/queue/rotational ("0" / "1")
  plus JUNIOR_GGUF as a declared absolute path with a declared size (the
  file is never opened or stat'ed), the hardware class and the bitnetd bind.
  1. Path: JUNIOR_GGUF must be set, absolute, end in .gguf and contain no
     ".." segment; otherwise hold.
  2. Mount: the longest mount point that prefixes the path wins.
  3. Filesystem: network or userspace filesystems (nfs, nfs4, cifs, smb3,
     9p, fuse.*) reject, because reading weights there is a fetch
     (fetch false). tmpfs/ramfs reject on SBC (weights would sit in RAM
     twice) and hold on SFF. vfat/exfat with a file over 4 GiB on vfat
     rejects (cannot hold it). Local fs (ext4, xfs, btrfs, f2fs, exfat)
     continue. Unknown fstype holds.
  4. Block: /dev/<part> maps to its parent disk (nvme0n1p2 -> nvme0n1,
     mmcblk0p2 -> mmcblk0, sda1 -> sda). removable=1 holds (the device can
     vanish under a live mmap; the operator copies to fixed storage, the bot
     does not). Unparsed sysfs values hold. Non-/dev sources on a local fs
     (overlay, etc.) hold.
  5. Load: cold-read seconds = size / class throughput of the disk kind
     (nvme 1500, sata-ssd 400, hdd 120, mmc 40 MB/s). Rotational disks get
     load mode "readahead", others "mmap". Over the class budget (SBC 120 s,
     SFF 60 s) still admits but flags slow_cold_load.
  6. Bind: bitnetd must be exactly 127.0.0.1:8765; anything else rejects.
  7. Exactness: a fixed ternary block (codes 0/1/2 = -1/0/+1, four per byte,
     low bits first) is packed, unpacked and dotted with int8 activations;
     the row passes only if that equals the scalar reference.
  One row per fixture is chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05, where pack is the 2-bit
  trit weight block plus trit codes for the verdict and bind.

What it does not touch
  No real /proc or /sys reads, no mount or umount, no file opened or stat'ed
  at JUNIOR_GGUF, no copy, no mmap, no model pull, no GGUF download, no
  sockets (nothing binds or connects), no device opened, no network fetch,
  no subprocess. mint/socket/layer_one/fetch/ue5_launch are all false.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_block_mount_junior_gguf_placement_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_block_mount_junior_gguf_placement_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

RATE = 0.05
GENESIS = "0" * 16
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
BITNETD_BIND = ("127.0.0.1", 8765)
GiB = 1024 ** 3
MB = 1000 ** 2

NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "9p", "sshfs"}
RAM_FS = {"tmpfs", "ramfs"}
LOCAL_FS = {"ext4", "xfs", "btrfs", "f2fs", "exfat", "vfat"}
THROUGHPUT_MBPS = {"nvme": 1500, "sata-ssd": 400, "hdd": 120, "mmc": 40}
LOAD_BUDGET_S = {"sbc": 120.0, "sff": 60.0}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


# --- ternary exactness block -------------------------------------------------
WEIGHTS = [1, -1, 0, 1, 0, 0, -1, 1, -1, 1, 1, 0, 0, -1, 1, -1]
ACTS = [3, -7, 12, 5, -2, 9, 4, -6, 1, 0, -11, 8, 2, 7, -3, 10]


def pack_trits(ws):
    out = bytearray()
    for i in range(0, len(ws), 4):
        b = 0
        for j, w in enumerate(ws[i:i + 4]):
            b |= (w + 1) << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_trits(buf, n):
    return [((buf[k // 4] >> (2 * (k % 4))) & 3) - 1 for k in range(n)]


def exact_ok():
    packed = pack_trits(WEIGHTS)
    got = sum(w * a for w, a in zip(unpack_trits(packed, len(WEIGHTS)), ACTS))
    ref = sum(w * a for w, a in zip(WEIGHTS, ACTS))
    return got == ref and len(packed) == len(WEIGHTS) * 2 // 8, packed


# --- parsing -----------------------------------------------------------------
def parse_mounts(text):
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        src, mnt, fstype, opts = parts[:4]
        rows.append({"src": src, "mnt": mnt, "fstype": fstype, "opts": opts.split(",")})
    return rows


def mount_for(path, mounts):
    best = None
    for m in mounts:
        mp = m["mnt"].rstrip("/") or "/"
        if path == mp or path.startswith(mp if mp == "/" else mp + "/"):
            if best is None or len(mp) > len(best["mnt"].rstrip("/") or "/"):
                best = m
    return best


def parent_disk(src):
    if not src.startswith("/dev/"):
        return None
    name = src[5:]
    m = re.fullmatch(r"(nvme\d+n\d+)(p\d+)?", name) or re.fullmatch(r"(mmcblk\d+)(p\d+)?", name) \
        or re.fullmatch(r"(sd[a-z]+)(\d+)?", name) or re.fullmatch(r"(vd[a-z]+)(\d+)?", name)
    return m.group(1) if m else None


def disk_kind(disk, rotational):
    if disk.startswith("nvme"):
        return "nvme"
    if disk.startswith("mmcblk"):
        return "mmc"
    return "hdd" if rotational else "sata-ssd"


def bit(v):
    v = (v or "").strip()
    return int(v) if v in ("0", "1") else None


# --- decision ----------------------------------------------------------------
def decide(fx):
    out = {"verdict": "hold", "reason": "", "load_mode": None, "cold_load_s": None, "slow_cold_load": False}
    if tuple(fx.get("bind", ())) != BITNETD_BIND:
        out.update(verdict="reject", reason="bitnetd bind is not exactly 127.0.0.1:8765")
        return out
    path = fx.get("junior_gguf")
    if not path or not path.startswith("/") or not path.endswith(".gguf") or "/../" in path + "/":
        out["reason"] = "JUNIOR_GGUF unset or not an absolute .gguf path; operator sets it"
        return out
    m = mount_for(path, parse_mounts(fx["mounts"]))
    if m is None:
        out["reason"] = "no mount covers JUNIOR_GGUF"
        return out
    fs = m["fstype"]
    cls = fx["class"]
    size = fx["size_bytes"]
    if fs in NETWORK_FS or fs.startswith("fuse"):
        out.update(verdict="reject", reason=f"{fs} is remote/userspace; reading weights there is a fetch")
        return out
    if fs in RAM_FS:
        if cls == "sbc":
            out.update(verdict="reject", reason=f"{fs} on SBC would hold weights in RAM twice")
        else:
            out["reason"] = f"{fs} on SFF: operator should place GGUF on fixed disk"
        return out
    if fs not in LOCAL_FS:
        out["reason"] = f"unknown fstype {fs}"
        return out
    if fs == "vfat" and size >= 4 * GiB:
        out.update(verdict="reject", reason="vfat cannot hold a file of 4 GiB or more")
        return out
    disk = parent_disk(m["src"])
    if disk is None:
        out["reason"] = f"source {m['src']} is not a plain block partition"
        return out
    sysfs = fx.get("sysfs", {}).get(disk, {})
    rem, rot = bit(sysfs.get("removable")), bit(sysfs.get("rotational"))
    if rem is None or rot is None:
        out["reason"] = f"sysfs for {disk} unparsed"
        return out
    if rem == 1:
        out["reason"] = f"{disk} is removable; operator copies GGUF to fixed storage, bot does not"
        return out
    kind = disk_kind(disk, rot)
    secs = round(size / (THROUGHPUT_MBPS[kind] * MB), 2)
    out.update(verdict="admit", reason=f"{fs} on fixed {kind} {disk}",
               load_mode="readahead" if rot else "mmap", cold_load_s=secs,
               slow_cold_load=secs > LOAD_BUDGET_S[cls])
    if "ro" in m["opts"]:
        out["reason"] += " (ro mount)"
    return out


MNT_BASE = "/dev/nvme0n1p2 / ext4 rw,relatime 0 0\nproc /proc proc rw 0 0\n"
FIXTURES = [
    {"name": "sff_nvme_ext4", "class": "sff", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/srv/models/bitnet-2b-i2s.gguf", "size_bytes": int(0.55 * GiB),
     "mounts": MNT_BASE, "sysfs": {"nvme0n1": {"removable": "0", "rotational": "0"}},
     "expect": ("admit", "mmap", False)},
    {"name": "sff_hdd_xfs_slow", "class": "sff", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/data/m/big-i2s.gguf", "size_bytes": 9 * GiB,
     "mounts": MNT_BASE + "/dev/sdb1 /data xfs ro,noatime 0 0\n",
     "sysfs": {"nvme0n1": {"removable": "0", "rotational": "0"}, "sdb": {"removable": "0", "rotational": "1"}},
     "expect": ("admit", "readahead", True)},
    {"name": "sbc_mmc_root", "class": "sbc", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/var/lib/bitnetd/bitnet-2b-i2s.gguf", "size_bytes": int(0.55 * GiB),
     "mounts": "/dev/mmcblk0p2 / ext4 rw 0 0\n", "sysfs": {"mmcblk0": {"removable": "0", "rotational": "0"}},
     "expect": ("admit", "mmap", False)},
    {"name": "sbc_usb_stick", "class": "sbc", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/media/stick/m.gguf", "size_bytes": GiB,
     "mounts": "/dev/mmcblk0p2 / ext4 rw 0 0\n/dev/sda1 /media/stick exfat rw 0 0\n",
     "sysfs": {"mmcblk0": {"removable": "0", "rotational": "0"}, "sda": {"removable": "1", "rotational": "0"}},
     "expect": ("hold", None, False)},
    {"name": "sff_nfs_share", "class": "sff", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/mnt/nas/m.gguf", "size_bytes": GiB,
     "mounts": MNT_BASE + "nas:/models /mnt/nas nfs4 ro 0 0\n", "sysfs": {},
     "expect": ("reject", None, False)},
    {"name": "sbc_tmpfs", "class": "sbc", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/run/m.gguf", "size_bytes": GiB,
     "mounts": "/dev/mmcblk0p2 / ext4 rw 0 0\ntmpfs /run tmpfs rw 0 0\n", "sysfs": {},
     "expect": ("reject", None, False)},
    {"name": "sff_vfat_too_big", "class": "sff", "bind": ("127.0.0.1", 8765),
     "junior_gguf": "/boot/efi/m.gguf", "size_bytes": 5 * GiB,
     "mounts": MNT_BASE + "/dev/nvme0n1p1 /boot/efi vfat rw 0 0\n",
     "sysfs": {"nvme0n1": {"removable": "0", "rotational": "0"}},
     "expect": ("reject", None, False)},
    {"name": "unset_env", "class": "sff", "bind": ("127.0.0.1", 8765),
     "junior_gguf": None, "size_bytes": 0, "mounts": MNT_BASE, "sysfs": {},
     "expect": ("hold", None, False)},
    {"name": "wildcard_bind", "class": "sff", "bind": ("0.0.0.0", 8765),
     "junior_gguf": "/srv/models/bitnet-2b-i2s.gguf", "size_bytes": GiB, "mounts": MNT_BASE,
     "sysfs": {"nvme0n1": {"removable": "0", "rotational": "0"}},
     "expect": ("reject", None, False)},
]

VERDICT_TRIT = {"reject": 0, "hold": 1, "admit": 2}


def run():
    ok_exact, wpack = exact_ok()
    rows, prev, all_ok = [], GENESIS, ok_exact
    for fx in FIXTURES:
        d = decide(fx)
        want_v, want_mode, want_slow = fx["expect"]
        passed = ok_exact and d["verdict"] == want_v and d["load_mode"] == want_mode and d["slow_cold_load"] == want_slow
        all_ok &= passed
        body = {"fixture": fx["name"], "class": fx["class"], **d, "pass": passed}
        text = json.dumps(body, sort_keys=True)
        bind_trit = 2 if tuple(fx["bind"]) == BITNETD_BIND else 0
        pack = wpack + bytes([VERDICT_TRIT[d["verdict"]] | (bind_trit << 2)])
        c = close(prev, pack, text)
        rows.append({"prev": prev, "body": body, "text": text, "pack": pack.hex(), "close": c})
        prev = c
    chain_ok, p = True, GENESIS
    for r in rows:
        chain_ok &= r["prev"] == p and close(p, bytes.fromhex(r["pack"]), r["text"]) == r["close"]
        p = r["close"]
    return {"seam": "block+mount -> JUNIOR_GGUF placement", "roadmap": "JuniorHome ROADMAP #2 GGUF",
            "rate": RATE, "flags": FLAGS, "exact": ok_exact, "chain_ok": chain_ok,
            "rows": len(rows), "pass": all_ok and chain_ok, "tip_close": prev,
            "results": [r["body"] for r in rows]}


if __name__ == "__main__":
    res = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in res["results"]:
            print(f"{r['fixture']:<18} {r['verdict']:<6} mode={r['load_mode']} cold={r['cold_load_s']} "
                  f"slow={r['slow_cold_load']} pass={r['pass']}  {r['reason']}")
        print("tip_close", res["tip_close"], "pass", res["pass"])
    else:
        print(json.dumps(res, indent=1))
    sys.exit(0 if res["pass"] else 1)
