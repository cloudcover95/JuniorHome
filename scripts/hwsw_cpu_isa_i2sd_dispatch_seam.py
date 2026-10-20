#!/usr/bin/env python3
"""Hardware-to-software seam emulator: CPU ISA features (/proc/cpuinfo + AT_HWCAP) -> i2sd userspace I2_S kernel pick.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace — 127.0.0.1:8767. No
  docker.sock.", on the hardware class rows from docs/HW_CLASS.md: SBC
  (arm64, <=12 W -> JuniorOS T4) and SFF host (x86_64 or arm64 -> JuniorHome
  T0). The seam answers one question: given what the CPU says it can do, which
  I2_S userspace kernel may i2sd pick, and how many threads? Rule: a SIMD path
  is allowed only when /proc/cpuinfo flags AND the auxv AT_HWCAP bits both
  report it (intersection); on disagreement i2sd downgrades, never upgrades.
  Whatever path is picked, the ternary dot product must equal the scalar
  reference exactly. Not ROADMAP T13 (overnight owns it).

What it emulates
  In-memory fixtures shaped like /proc/cpuinfo text (x86 "flags", arm64
  "Features"), an AT_HWCAP integer, a core count, a watt class and the i2sd
  bind string.
  1. ISA parse: x86_64 flags -> {avx512bw, avx512_vnni, avx2, ssse3};
     arm64 Features -> {asimd, asimddp, sve}. arm64 AT_HWCAP bits use the
     Linux uapi values ASIMD 1<<1, ASIMDDP 1<<20, SVE 1<<22.
  2. Agreement: arm64 features are intersected with AT_HWCAP; a feature in
     cpuinfo but missing from hwcap (or the reverse) is dropped and the row
     records a downgrade. x86 has no AT_HWCAP vector bits, so cpuinfo flags
     alone are used (AT_HWCAP there is the edx word and is ignored).
  3. Kernel ladder (first match): x86 avx512_vnni+avx512bw -> i2s_avx512vnni,
     avx2 -> i2s_avx2, ssse3 -> i2s_ssse3; arm64 asimddp -> i2s_neon_dot,
     asimd -> i2s_neon; else i2s_scalar. SVE is seen but not used.
  4. Threads: SBC class (<=12 W) caps at 4, SFF host caps at 8, never more
     than the core count, never below 1.
  5. Bind: i2sd must be exactly 127.0.0.1:8767; anything else rejects the row.
  6. Exactness: a fixed ternary weight block (codes 0/1/2 = -1/0/+1, four per
     byte, I2_S order, low bits first) is dotted with int8 activations by two
     stdlib routes, a per-byte lookup table and a plain unpack loop. Both
     stand in for the chosen kernel; the row passes only if both equal the
     scalar reference, so a kernel pick can never change the answer.
  One row per host fixture is chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05, where pack is the real
  2-bit trit weight block plus trit codes for agreement and bind state.

What it does not touch
  No /proc, /sys, auxv, cpuid instruction, ctypes, or compiled kernel; no
  rails/linux/i2s_pack.c build. No socket is opened: the bind is a string and
  is never bound. It does not import or re-run i2sd, bitnetd, juniorctl,
  scripts/advance_*, ports/*, OSai bench notes, or any other team file, and it
  does not cover the earlier 06:00 seams (spool, USB sound/HID, SBC T4, SFF
  T0, USB-C, tun). No model pull, no GGUF. Rows go to a tempfile removed
  after the run. Standard library only.
  Flags: mint, socket, layer_one, fetch, ue5_launch False.

How to run
  python3 scripts/hwsw_cpu_isa_i2sd_dispatch_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_cpu_isa_i2sd_dispatch_seam.py demo     # emulated host rows
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
I2SD_BIND = "127.0.0.1:8767"
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16

ARM64_HWCAP = {"asimd": 1 << 1, "asimddp": 1 << 20, "sve": 1 << 22}
X86_SEEN = ("avx512bw", "avx512_vnni", "avx2", "ssse3")
THREAD_CAP = {"sbc": 4, "sff": 8}

# fixed weight codes (0 -> -1, 1 -> 0, 2 -> +1) and int8 activations
W_CODES = [2, 0, 1, 2, 1, 1, 0, 2, 2, 2, 0, 1, 1, 0, 2, 0,
           0, 1, 2, 2, 1, 0, 0, 2, 2, 1, 1, 1, 0, 2, 1, 0]
ACTS = [17, -3, 88, -128, 5, 127, -64, 9, 0, -1, 33, 71, -90, 12, 4, -7,
        64, -5, 2, 19, -45, 100, 8, -16, 55, 3, -127, 21, 6, -2, 70, 11]


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes: list[int]) -> bytes:
    codes = list(codes)
    while len(codes) % 4:
        codes.append(1)  # pad with the zero weight
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            assert c in (0, 1, 2)
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def unpack2(pack: bytes, n: int) -> list[int]:
    return [(pack[k // 4] >> (2 * (k % 4))) & 3 for k in range(n)]


def dot_scalar(codes: list[int], acts: list[int]) -> int:
    return sum((c - 1) * a for c, a in zip(codes, acts))


def dot_unpack(pack: bytes, acts: list[int]) -> int:
    return sum((c - 1) * a for c, a in zip(unpack2(pack, len(acts)), acts))


_LUT = [tuple(((b >> (2 * j)) & 3) - 1 for j in range(4)) for b in range(256)]


def dot_lut(pack: bytes, acts: list[int]) -> int:
    total = 0
    for i, b in enumerate(pack):
        w = _LUT[b]
        for j in range(4):
            k = 4 * i + j
            if k < len(acts):
                total += w[j] * acts[k]
    return total


def parse_cpuinfo(text: str) -> tuple[str, set[str]]:
    arch, feats = "unknown", set()
    for line in text.splitlines():
        key, sep, val = line.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        if key == "flags":
            arch, feats = "x86_64", set(val.split())
        elif key == "features":
            arch, feats = "arm64", set(val.split())
    return arch, feats


def isa(text: str, hwcap: int) -> dict:
    arch, raw = parse_cpuinfo(text)
    dropped: list[str] = []
    if arch == "arm64":
        seen = {f for f in ARM64_HWCAP if f in raw}
        from_hwcap = {f for f, bit in ARM64_HWCAP.items() if hwcap & bit}
        usable = seen & from_hwcap
        dropped = sorted(seen ^ from_hwcap)
    elif arch == "x86_64":
        usable = {f for f in X86_SEEN if f in raw}
    else:
        usable = set()
    return {"arch": arch, "usable": sorted(usable), "dropped": dropped,
            "agree": not dropped}


def pick_kernel(arch: str, usable: list[str]) -> str:
    u = set(usable)
    if arch == "x86_64":
        if {"avx512_vnni", "avx512bw"} <= u:
            return "i2s_avx512vnni"
        if "avx2" in u:
            return "i2s_avx2"
        if "ssse3" in u:
            return "i2s_ssse3"
    elif arch == "arm64":
        if "asimddp" in u:
            return "i2s_neon_dot"
        if "asimd" in u:
            return "i2s_neon"
    return "i2s_scalar"


def threads(cls: str, cores: int) -> int:
    return max(1, min(int(cores), THREAD_CAP.get(cls, 1)))


def build_row(prev: str, fx: dict) -> dict:
    info = isa(fx["cpuinfo"], fx.get("hwcap", 0))
    kernel = pick_kernel(info["arch"], info["usable"])
    bind_ok = fx["bind"] == I2SD_BIND
    wpack = pack2(W_CODES)
    ref = dot_scalar(W_CODES, ACTS)
    exact = dot_lut(wpack, ACTS) == ref and dot_unpack(wpack, ACTS) == ref
    if not bind_ok:
        state, why = "reject", "i2sd_bind_not_loopback_8767"
    elif not exact:
        state, why = "reject", "kernel_result_mismatch"
    elif not info["agree"]:
        state, why = "ready_downgraded", "hwcap_cpuinfo_disagree"
    else:
        state, why = "ready", ""
    body = {"host": fx["host"], "class": fx["class"], "isa": info,
            "kernel": kernel, "threads": threads(fx["class"], fx["cores"]),
            "bind": fx["bind"], "bind_ok": bind_ok, "dot_ref": ref,
            "exact": exact, "state": state, "why": why,
            "uses_sve": False, "rate": RATE, "flags": FLAGS}
    status = [0 if info["agree"] else 1, 0 if bind_ok else 2,
              {"ready": 0, "ready_downgraded": 1, "reject": 2}[state]]
    pack = wpack + pack2(status)
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {"body": body, "pack": pack.hex(), "prev": prev,
            "close": close(prev, pack, text)}


FIXTURES = [
    {"host": "sff_x86_avx2", "class": "sff", "cores": 12, "hwcap": 0xbfebfbff,
     "cpuinfo": "processor\t: 0\nflags\t\t: fpu sse2 ssse3 sse4_1 sse4_2 avx avx2 fma bmi2\n",
     "bind": I2SD_BIND, "want": ("ready", "i2s_avx2", 8)},
    {"host": "sff_x86_avx512vnni", "class": "sff", "cores": 6, "hwcap": 0xbfebfbff,
     "cpuinfo": "flags\t\t: sse2 ssse3 avx2 avx512f avx512bw avx512_vnni\n",
     "bind": I2SD_BIND, "want": ("ready", "i2s_avx512vnni", 6)},
    {"host": "sbc_arm64_dotprod", "class": "sbc", "cores": 8,
     "hwcap": (1 << 0) | (1 << 1) | (1 << 20),
     "cpuinfo": "processor\t: 0\nFeatures\t: fp asimd evtstrm aes crc32 asimddp\n",
     "bind": I2SD_BIND, "want": ("ready", "i2s_neon_dot", 4)},
    {"host": "sbc_arm64_cpuinfo_claims_dp_hwcap_not", "class": "sbc", "cores": 4,
     "hwcap": (1 << 0) | (1 << 1),
     "cpuinfo": "Features\t: fp asimd asimddp\n",
     "bind": I2SD_BIND, "want": ("ready_downgraded", "i2s_neon", 4)},
    {"host": "sff_arm64_sve_seen_not_used", "class": "sff", "cores": 16,
     "hwcap": (1 << 1) | (1 << 20) | (1 << 22),
     "cpuinfo": "Features\t: fp asimd asimddp sve\n",
     "bind": I2SD_BIND, "want": ("ready", "i2s_neon_dot", 8)},
    {"host": "sbc_old_core_scalar", "class": "sbc", "cores": 1, "hwcap": 0,
     "cpuinfo": "Features\t: fp\n",
     "bind": I2SD_BIND, "want": ("ready", "i2s_scalar", 1)},
    {"host": "sff_wildcard_bind", "class": "sff", "cores": 8, "hwcap": 0,
     "cpuinfo": "flags\t\t: sse2 ssse3 avx2\n",
     "bind": "0.0.0.0:8767", "want": ("reject", "i2s_avx2", 8)},
]


def run_rows() -> list[dict]:
    rows, prev = [], GENESIS
    for fx in FIXTURES:
        r = build_row(prev, fx)
        rows.append(r)
        prev = r["close"]
    return rows


def self_test() -> dict:
    rows = run_rows()
    results = []
    for fx, r in zip(FIXTURES, rows):
        b = r["body"]
        ok = (b["state"], b["kernel"], b["threads"]) == fx["want"] and b["exact"]
        ok = ok and b["uses_sve"] is False
        n = len(W_CODES)
        ok = ok and unpack2(bytes.fromhex(r["pack"]), n) == W_CODES
        results.append({"host": fx["host"], "state": b["state"], "kernel": b["kernel"],
                        "threads": b["threads"], "ok": ok})
    prev, chain_ok = GENESIS, True
    for r in rows:
        text = json.dumps(r["body"], sort_keys=True, separators=(",", ":"))
        chain_ok &= r["prev"] == prev and close(prev, bytes.fromhex(r["pack"]), text) == r["close"]
        prev = r["close"]
    fd, path = tempfile.mkstemp(prefix="hwsw_isa_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        with open(path) as fh:
            disk_ok = [json.loads(line)["close"] for line in fh] == [r["close"] for r in rows]
    finally:
        os.remove(path)
    flags_ok = all(v is False for v in FLAGS.values())
    passed = all(x["ok"] for x in results) and chain_ok and disk_ok and flags_ok
    return {"seam": "cpu_isa(cpuinfo+hwcap)->i2sd_i2s_kernel_pick", "rate": RATE,
            "dot_ref": dot_scalar(W_CODES, ACTS), "rows": results,
            "chain_ok": chain_ok, "disk_ok": disk_ok, "flags": FLAGS,
            "tip_close": rows[-1]["close"], "pass": passed}


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "demo":
        for r in run_rows():
            b = r["body"]
            print(json.dumps({"host": b["host"], "kernel": b["kernel"], "threads": b["threads"],
                              "state": b["state"], "why": b["why"], "close": r["close"]},
                             sort_keys=True))
        return 0
    res = self_test()
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
