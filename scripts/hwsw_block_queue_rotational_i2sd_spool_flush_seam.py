#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/block/*/queue attributes -> i2sd receipt-spool flush mode.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where the spool disk's block-queue attributes
  (rotational, scheduler, write_cache) choose how i2sd flushes its receipt
  spool. Hardware classes: SBC arm64 (JuniorOS T4, SD/eMMC/NVMe) and SFF host
  (JuniorHome T0, spool on SSD/HDD). Not T13, not item 2 (no GGUF), not item 4
  (no Asahi/MLX), not item 5. No other team file or earlier hwsw_* seam is
  imported, read, or wrapped; block queue attributes were not covered before.

What it emulates
  In-memory fixtures shaped like /sys/block/<dev>/queue files: rotational
  (0/1), scheduler (text with the active entry in [brackets]), write_cache
  ("write back" / "write through"), read_only (0/1). Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. rotational/read_only not 0/1, no bracketed scheduler, or unknown
     write_cache -> hold (unparsed).
  4. read_only=1 -> hold (ro_spool): spool cannot be written.
  5. Rotational disk -> soft (hdd_batch): batch flush, 64 receipts per fsync.
  6. Write-back cache with scheduler "none" or "mq-deadline" -> admit
     (fsync_16): 16 receipts per fsync.
  7. Write-through, or other scheduler -> admit (fsync_4): 4 per fsync.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no block device opened, no fsync or sync call,
  no udev, no systemd, no sockets (binds are strings), no network fetch, no
  subprocess, no model pull, no GGUF download. Rows live in memory only.
  Flags mint, socket, layer_one, fetch, ue5_launch all False. JuniorSolana and
  the profile repo are untouched.

How to run
  python3 scripts/hwsw_block_queue_rotational_i2sd_spool_flush_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_block_queue_rotational_i2sd_spool_flush_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8767"
CLASSES = ("sbc", "sff")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}
CACHES = ("write back", "write through")


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


def active_sched(text):
    m = re.search(r"\[([A-Za-z0-9_-]+)\]", text if isinstance(text, str) else "")
    return m.group(1) if m else None


def decide(fx):
    """Return (decision, reason, batch, codes)."""
    if fx.get("bind") != BIND_OK:
        return "reject", "bind_reject", 0, [0]
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0, [2, 0]
    if fx.get("cls") not in CLASSES:
        return "hold", "bad_class", 0, [2, 2, 0]
    rot, ro = fx.get("rotational"), fx.get("read_only")
    sched, wc = active_sched(fx.get("scheduler")), fx.get("write_cache")
    if rot not in (0, 1) or ro not in (0, 1) or sched is None or wc not in CACHES:
        return "hold", "unparsed", 0, [2, 2, 2, 0]
    if ro == 1:
        return "hold", "ro_spool", 0, [2, 2, 2, 0]
    if rot == 1:
        return "soft", "hdd_batch", 64, [2, 2, 2, 1]
    if wc == "write back" and sched in ("none", "mq-deadline"):
        return "admit", "fsync_16", 16, [2, 2, 2, 2]
    return "admit", "fsync_4", 4, [2, 2, 2, 2]


def Q(rot, sched, wc, ro=0):
    return dict(rotational=rot, scheduler=sched, write_cache=wc, read_only=ro)


FIXTURES = [
    ("sbc_nvme_none", dict(cls="sbc", bind=BIND_OK, **Q(0, "[none] mq-deadline kyber", "write back")), "admit", 16),
    ("sff_ssd_mqdl", dict(cls="sff", bind=BIND_OK, **Q(0, "none [mq-deadline] bfq", "write back")), "admit", 16),
    ("sbc_emmc_bfq", dict(cls="sbc", bind=BIND_OK, **Q(0, "none mq-deadline [bfq]", "write back")), "admit", 4),
    ("sff_wt_cache", dict(cls="sff", bind=BIND_OK, **Q(0, "[none]", "write through")), "admit", 4),
    ("sff_hdd", dict(cls="sff", bind=BIND_OK, **Q(1, "none [mq-deadline]", "write back")), "soft", 64),
    ("sbc_ro_sd", dict(cls="sbc", bind=BIND_OK, **Q(0, "[none]", "write through", 1)), "hold", 0),
    ("no_active_sched", dict(cls="sff", bind=BIND_OK, **Q(0, "none mq-deadline", "write back")), "hold", 0),
    ("bad_rotational", dict(cls="sff", bind=BIND_OK, **Q(3, "[none]", "write back")), "hold", 0),
    ("bad_class", dict(cls="x86mac", bind=BIND_OK, **Q(0, "[none]", "write back")), "hold", 0),
    ("bind_any", dict(cls="sff", bind="0.0.0.0:8767", **Q(0, "[none]", "write back")), "reject", 0),
    ("docker_sock", dict(cls="sff", bind=BIND_OK, docker_sock=True, **Q(0, "[none]", "write back")), "reject", 0),
]


def run():
    prev, rows, fails = GENESIS, [], []
    for name, fx, want_d, want_b in FIXTURES:
        d, reason, b, codes = decide(fx)
        pk = pack2(codes)
        prev = close(prev, pk, "%s|%s|%s|%d" % (name, d, reason, b))
        rows.append({"name": name, "decision": d, "reason": reason, "batch": b, "pack": pk.hex(), "close": prev})
        if (d, b) != (want_d, want_b):
            fails.append(name)
    pack_ok = pack2([2, 1, 0, 2]) == bytes([2 | (1 << 2) | (0 << 4) | (2 << 6)])
    return rows, fails, pack_ok, prev


if __name__ == "__main__":
    rows, fails, pack_ok, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print("%-16s %-6s %-12s batch=%d pack=%s close=%s" % (r["name"], r["decision"], r["reason"], r["batch"], r["pack"], r["close"]))
    ok = not fails and pack_ok and not any(FLAGS.values())
    print(json.dumps({"rate": RATE, "bind": BIND_OK, "flags": FLAGS, "rows": len(rows),
                      "pack_order_ok": pack_ok, "fail": fails, "pass": ok, "tip_close": tip}))
    sys.exit(0 if ok else 1)
