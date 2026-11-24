#!/usr/bin/env python3
"""Hardware-to-software seam emulator: eMMC/SD wear attributes -> i2sd spool write budget.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace - 127.0.0.1:8767. No
  docker.sock.", at the seam where flash wear (eMMC life_time / pre_eol, as
  exposed under /sys/block/mmcblk*/device) decides how much spool writing
  i2sd may do. Hardware classes: SBC arm64 (JuniorOS T4) and SFF host
  (JuniorHome T0, usually NVMe/SATA so no eMMC attributes is normal).
  Not T13, not item 2 (no GGUF), not item 4 (no Asahi/MLX), not item 5.
  No other team file or earlier hwsw_* seam is imported, read, or wrapped.

What it emulates
  In-memory fixtures shaped like eMMC sysfs attributes: life_time as the
  kernel prints it ("0x01 0x02", two 10%-step estimates, 0x01 = 0-10% used,
  0x0b = exceeded), pre_eol (0x01 normal, 0x02 warning, 0x03 urgent).
  Rules, in order:
  1. Bind must be exactly 127.0.0.1:8767, else reject (bind_reject).
     docker_sock True -> reject (docker_sock).
  2. Class must be "sbc" or "sff", else hold (bad_class).
  3. SFF with no wear attributes -> admit full budget (no_emmc).
     SBC with none -> soft, half budget (no_wear_info).
  4. Unparseable life_time / pre_eol -> hold (unparsed).
  5. pre_eol 0x03 -> hold (urgent): spool writes stop, i2sd read-only.
  6. worst life_time step >= 0x0a or pre_eol 0x02 -> soft (worn): 1/4 budget.
  7. worst step >= 0x07 -> soft: 1/2 budget.
  8. Otherwise admit, full budget.
  Full budget is wanted MiB/day capped at SBC 256, SFF 2048.
  Rows pack decision and check states as 2-bit trit codes (four per byte,
  I2_S order, first code in low bits) and chain with close rule
  sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no writes, no /dev nodes, no mmc-utils or
  smartctl binary, no systemd, no sockets (binds are strings), no device
  opened, no network fetch, no subprocess, no model pull, no GGUF download.
  Rows live in memory only. Flags mint, socket, layer_one, fetch,
  ue5_launch all False. JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_emmc_wear_i2sd_spool_budget_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_emmc_wear_i2sd_spool_budget_seam.py demo   # one line per fixture row
"""
import hashlib
import json
import sys

BIND = "127.0.0.1:8767"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
CAP = {"sbc": 256, "sff": 2048}
TRIT = {"admit": 1, "soft": 0, "hold": 2, "reject": 2}


def parse_hex(tok):
    try:
        if not tok.startswith("0x"):
            return None
        return int(tok, 16)
    except (ValueError, AttributeError):
        return None


def decide(fx):
    if fx.get("bind") != BIND:
        return "reject", "bind_reject", 0
    if fx.get("docker_sock"):
        return "reject", "docker_sock", 0
    cls = fx.get("cls")
    if cls not in CAP:
        return "hold", "bad_class", 0
    full = min(int(fx.get("wanted_mib_day", CAP[cls])), CAP[cls])
    lt, eol = fx.get("life_time"), fx.get("pre_eol")
    if lt is None and eol is None:
        if cls == "sff":
            return "admit", "no_emmc", full
        return "soft", "no_wear_info", full // 2
    steps = []
    if lt is not None:
        parts = str(lt).split()
        steps = [parse_hex(p) for p in parts]
        if len(parts) != 2 or any(s is None or s > 0x0b for s in steps):
            return "hold", "unparsed", 0
    e = None
    if eol is not None:
        e = parse_hex(str(eol).strip())
        if e is None or e not in (1, 2, 3):
            return "hold", "unparsed", 0
    if e == 3:
        return "hold", "urgent", 0
    worst = max(steps) if steps else 0
    if worst >= 0x0a or e == 2:
        return "soft", "worn", full // 4
    if worst >= 0x07:
        return "soft", "half_life", full // 2
    return "admit", "ok", full


def close(prev, pack, text):
    h = hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()
    return h[:16]


def pack_trits(codes):
    out = bytearray((len(codes) + 3) // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= (c & 3) << (2 * (i % 4))
    return bytes(out)


FIXTURES = [
    ("sbc_new", {"cls": "sbc", "bind": BIND, "life_time": "0x01 0x01", "pre_eol": "0x01"}, ("admit", "ok", 256)),
    ("sbc_half", {"cls": "sbc", "bind": BIND, "life_time": "0x07 0x03", "pre_eol": "0x01"}, ("soft", "half_life", 128)),
    ("sbc_worn", {"cls": "sbc", "bind": BIND, "life_time": "0x0a 0x02", "pre_eol": "0x01"}, ("soft", "worn", 64)),
    ("sbc_eol_warn", {"cls": "sbc", "bind": BIND, "life_time": "0x02 0x02", "pre_eol": "0x02"}, ("soft", "worn", 64)),
    ("sbc_urgent", {"cls": "sbc", "bind": BIND, "life_time": "0x0b 0x0b", "pre_eol": "0x03"}, ("hold", "urgent", 0)),
    ("sbc_garbage", {"cls": "sbc", "bind": BIND, "life_time": "wear?", "pre_eol": "0x01"}, ("hold", "unparsed", 0)),
    ("sbc_out_of_range", {"cls": "sbc", "bind": BIND, "life_time": "0x0c 0x01"}, ("hold", "unparsed", 0)),
    ("sbc_none", {"cls": "sbc", "bind": BIND}, ("soft", "no_wear_info", 128)),
    ("sff_nvme_none", {"cls": "sff", "bind": BIND, "wanted_mib_day": 5000}, ("admit", "no_emmc", 2048)),
    ("sff_small_want", {"cls": "sff", "bind": BIND, "wanted_mib_day": 100}, ("admit", "no_emmc", 100)),
    ("bad_class", {"cls": "tower", "bind": BIND}, ("hold", "bad_class", 0)),
    ("bad_bind", {"cls": "sbc", "bind": "0.0.0.0:8767"}, ("reject", "bind_reject", 0)),
    ("docker_sock", {"cls": "sff", "bind": BIND, "docker_sock": True}, ("reject", "docker_sock", 0)),
]


def run():
    prev = "0" * 16
    rows, ok = [], True
    for name, fx, want in FIXTURES:
        got = decide(fx)
        passed = got == want
        ok = ok and passed
        pack = pack_trits([TRIT[got[0]], 1 if passed else 2])
        prev = close(prev, pack, name + ":" + got[0] + ":" + got[1])
        rows.append({"row": name, "got": list(got), "pass": passed, "close": prev})
    return ok, rows, prev


def main():
    ok, rows, tip = run()
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        for r in rows:
            print(r["row"], r["got"], "PASS" if r["pass"] else "FAIL", r["close"])
    else:
        print(json.dumps({"bind": BIND, "rate": RATE, "flags": FLAGS, "rows": len(rows),
                          "passed": sum(r["pass"] for r in rows), "tip_close": tip, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
