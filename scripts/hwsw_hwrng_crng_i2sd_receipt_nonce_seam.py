#!/usr/bin/env python3
"""Hardware-to-software seam emulator: hw_random + kernel CRNG -> i2sd receipt nonce admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767. No
  docker.sock." The seam decides, from the entropy layer alone, whether the
  i2sd userspace daemon may draw a per-receipt nonce for an I2_S work
  receipt ("ready"), may draw one but must flag it as kernel-CSPRNG-only or
  low-quality ("soft"), or must hold until the kernel random pool is
  initialised ("hold"). Battery-less SBC boards (JuniorOS T4) often boot
  with no hardware RNG and a slow CRNG init; SFF hosts (JuniorHome T0)
  usually expose a CPU RNG instruction or a TPM-backed hw_random.
  Not ROADMAP T13 (overnight owns it). Not item 5 (ue5_launch stays False).

What it emulates
  In-memory fixtures shaped like /sys/class/misc/hw_random/{rng_current,
  rng_available,rng_quality}, /proc/sys/kernel/random/{entropy_avail,
  poolsize}, a crng_ready flag (what getrandom(GRND_NONBLOCK) returning data
  instead of EAGAIN would tell you), a CPU RNG flag (x86 rdrand/rdseed in
  /proc/cpuinfo, or arm64 HWCAP2_RNG), and a 16-byte nonce sample. Rules,
  applied in order:
  1. Bind: the i2sd bind string must be exactly 127.0.0.1:8767; anything
     else (0.0.0.0, another port) -> "reject".
  2. poolsize must parse and be 256 (kernel >= 5.18) or 4096 (older);
     entropy_avail must parse. Otherwise -> "hold" (pool_unparsed).
  3. crng_ready False -> "hold" (crng_not_ready). The seam never blocks on
     /dev/random; the daemon retries later.
  4. Nonce sample all-zero, or identical to an earlier row's sample in the
     same run -> "hold" (nonce_zero / nonce_repeat).
  5. Hardware source: rng_current not "none" and listed in rng_available
     with rng_quality >= 512 -> strong; rng_current listed with quality
     1..511 -> "soft" (hwrng_low_quality); rng_current not in rng_available
     -> "hold" (hwrng_mismatch). No hw_random (absent or "none"): a CPU RNG
     flag counts as strong; otherwise SBC -> "soft" (no_hwrng) and SFF ->
     "soft" too but flagged sff_no_hwrng so the audit sees it.
  6. On a 4096-bit pool (pre-5.18 kernel), entropy_avail < 256 downgrades
     "ready" to "soft" (pool_low). On a 256 pool the value is informational.
  Each row packs its decision and three check states (bind, crng, hw
  source) as real 2-bit trit codes (hold/reject 0, soft 1, ready 2; four
  per byte, I2_S order: first code in the low bits), and rows are chained
  with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys, /proc, or /dev reads, no getrandom() or os.urandom() call,
  no rngd, no TPM, no ioctl, no device opened, no sockets (binds are
  strings; nothing listens or connects), no model pull, no GGUF, no network
  fetch, no subprocess. It does not import or re-run any team file or any
  earlier hwsw_* seam (RTC, watchdog, power, thermal, CPU ISA, memory,
  storage, USB-C, and so on); it has its own small trit packer. Rows go to
  a tempfile removed after the run; nothing is written under ~/.juniorhome.
  Flags: mint, socket, layer_one, fetch, ue5_launch all False. JuniorSolana
  and the profile repo are untouched.

How to run
  python3 scripts/hwsw_hwrng_crng_i2sd_receipt_nonce_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_hwrng_crng_i2sd_receipt_nonce_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

BIND = "127.0.0.1:8767"
RATE = 0.05
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
GENESIS = "0" * 16
CODE = {"reject": 0, "hold": 0, "soft": 1, "ready": 2}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack_trits(codes):
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            if c not in (0, 1, 2):
                raise ValueError("trit code must be 0, 1 or 2")
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_trits(data: bytes, n: int):
    codes = []
    for b in data:
        for j in range(4):
            codes.append((b >> (2 * j)) & 3)
    return codes[:n]


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def decide(fx, seen_nonces):
    """Return (decision, flags, trit codes) for one emulated host."""
    flags = []
    if fx.get("bind") != BIND:
        return "reject", ["bind_not_loopback_8767"], [0, 0, 0, 0]
    bind_c = 2
    pool = _int(fx.get("poolsize"))
    avail = _int(fx.get("entropy_avail"))
    if pool not in (256, 4096) or avail is None:
        return "hold", ["pool_unparsed"], [0, bind_c, 0, 0]
    if not fx.get("crng_ready"):
        return "hold", ["crng_not_ready"], [0, bind_c, 0, 0]
    crng_c = 2
    nonce = fx.get("nonce_hex", "")
    try:
        nb = bytes.fromhex(nonce)
    except ValueError:
        nb = b""
    if len(nb) != 16 or not any(nb):
        return "hold", ["nonce_zero"], [0, bind_c, crng_c, 0]
    if nb in seen_nonces:
        return "hold", ["nonce_repeat"], [0, bind_c, crng_c, 0]
    seen_nonces.add(nb)

    cur = (fx.get("rng_current") or "none").strip()
    avail_list = (fx.get("rng_available") or "").split()
    quality = _int(fx.get("rng_quality"))
    if cur != "none":
        if cur not in avail_list:
            return "hold", ["hwrng_mismatch"], [0, bind_c, crng_c, 0]
        if quality is not None and quality >= 512:
            dec, hw_c = "ready", 2
        else:
            dec, hw_c = "soft", 1
            flags.append("hwrng_low_quality")
    elif fx.get("cpu_rng"):
        dec, hw_c = "ready", 2
        flags.append("cpu_rng")
    else:
        dec, hw_c = "soft", 1
        flags.append("no_hwrng" if fx["cls"] == "sbc" else "sff_no_hwrng")
    if pool == 4096 and avail < 256 and dec == "ready":
        dec = "soft"
        hw_c = 1
        flags.append("pool_low")
    return dec, flags, [CODE[dec], bind_c, crng_c, hw_c]


FIXTURES = [
    ("sbc_bcm_hwrng_ready", {"cls": "sbc", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                             "crng_ready": True, "rng_current": "bcm2835-rng", "rng_available": "bcm2835-rng",
                             "rng_quality": "1000", "cpu_rng": False, "nonce_hex": "1f" * 16},
     "ready", []),
    ("sbc_no_hwrng_soft", {"cls": "sbc", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                           "crng_ready": True, "rng_current": "none", "rng_available": "",
                           "rng_quality": "", "cpu_rng": False, "nonce_hex": "a5" * 8 + "5a" * 8},
     "soft", ["no_hwrng"]),
    ("sbc_crng_cold_hold", {"cls": "sbc", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                            "crng_ready": False, "rng_current": "none", "rng_available": "",
                            "rng_quality": "", "cpu_rng": False, "nonce_hex": "33" * 16},
     "hold", ["crng_not_ready"]),
    ("sff_tpm_ready", {"cls": "sff", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                       "crng_ready": True, "rng_current": "tpm-rng-0", "rng_available": "tpm-rng-0",
                       "rng_quality": "1024", "cpu_rng": True, "nonce_hex": "0123456789abcdef" * 2},
     "ready", []),
    ("sff_cpu_rng_ready", {"cls": "sff", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                           "crng_ready": True, "rng_current": "none", "rng_available": "",
                           "rng_quality": "", "cpu_rng": True, "nonce_hex": "fedcba9876543210" * 2},
     "ready", ["cpu_rng"]),
    ("sff_old_pool_low_soft", {"cls": "sff", "bind": BIND, "poolsize": "4096", "entropy_avail": "128",
                               "crng_ready": True, "rng_current": "none", "rng_available": "",
                               "rng_quality": "", "cpu_rng": True, "nonce_hex": "77" * 15 + "01"},
     "soft", ["cpu_rng", "pool_low"]),
    ("sbc_low_quality_soft", {"cls": "sbc", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                              "crng_ready": True, "rng_current": "rockchip-rng", "rng_available": "rockchip-rng",
                              "rng_quality": "300", "cpu_rng": False, "nonce_hex": "c3" * 16},
     "soft", ["hwrng_low_quality"]),
    ("sbc_hwrng_mismatch_hold", {"cls": "sbc", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                                 "crng_ready": True, "rng_current": "iproc-rng200", "rng_available": "bcm2835-rng",
                                 "rng_quality": "1000", "cpu_rng": False, "nonce_hex": "e1" * 16},
     "hold", ["hwrng_mismatch"]),
    ("sff_nonce_repeat_hold", {"cls": "sff", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                               "crng_ready": True, "rng_current": "tpm-rng-0", "rng_available": "tpm-rng-0",
                               "rng_quality": "1024", "cpu_rng": True, "nonce_hex": "1f" * 16},
     "hold", ["nonce_repeat"]),
    ("sff_zero_nonce_hold", {"cls": "sff", "bind": BIND, "poolsize": "256", "entropy_avail": "256",
                             "crng_ready": True, "rng_current": "none", "rng_available": "",
                             "rng_quality": "", "cpu_rng": True, "nonce_hex": "00" * 16},
     "hold", ["nonce_zero"]),
    ("sbc_pool_unparsed_hold", {"cls": "sbc", "bind": BIND, "poolsize": "512", "entropy_avail": "256",
                                "crng_ready": True, "rng_current": "none", "rng_available": "",
                                "rng_quality": "", "cpu_rng": False, "nonce_hex": "44" * 16},
     "hold", ["pool_unparsed"]),
    ("sff_wildcard_bind_reject", {"cls": "sff", "bind": "0.0.0.0:8767", "poolsize": "256", "entropy_avail": "256",
                                  "crng_ready": True, "rng_current": "tpm-rng-0", "rng_available": "tpm-rng-0",
                                  "rng_quality": "1024", "cpu_rng": True, "nonce_hex": "55" * 16},
     "reject", ["bind_not_loopback_8767"]),
]


def _text(name, cls, dec, flags):
    return json.dumps({"row": name, "cls": cls, "decision": dec, "flags": flags, "rate": RATE}, sort_keys=True)


def run(verbose=False):
    failures = []
    rows = []
    prev = GENESIS
    seen = set()
    fd, path = tempfile.mkstemp(prefix="hwsw_hwrng_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for name, fx, want_dec, want_flags in FIXTURES:
                dec, flags, codes = decide(dict(fx), seen)
                pack = pack_trits(codes)
                if unpack_trits(pack, len(codes)) != codes:
                    failures.append(name + ":pack_roundtrip")
                cur = close(prev, pack, _text(name, fx["cls"], dec, flags))
                row = {"row": name, "decision": dec, "flags": flags, "pack": pack.hex(), "prev": prev, "close": cur}
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
                if dec != want_dec or flags != want_flags:
                    failures.append("%s: got %s %s want %s %s" % (name, dec, flags, want_dec, want_flags))
                if verbose:
                    print(json.dumps(row, sort_keys=True))
                prev = cur
        with open(path) as fh:
            if sum(1 for _ in fh) != len(FIXTURES):
                failures.append("tempfile_row_count")
    finally:
        os.remove(path)
    # Recompute the chain from scratch to confirm the close rule is deterministic.
    p = GENESIS
    seen2 = set()
    for (name, fx, _, _), r in zip(FIXTURES, rows):
        dec, flags, codes = decide(dict(fx), seen2)
        p2 = close(p, pack_trits(codes), _text(name, fx["cls"], dec, flags))
        if p2 != r["close"]:
            failures.append(name + ":chain")
        p = p2
    if any(FLAGS.values()):
        failures.append("flags_not_false")
    return {"seam": "hwrng_crng->i2sd_receipt_nonce", "roadmap": "ROADMAP #3 i2sd userspace", "bind": BIND,
            "rate": RATE, "flags": FLAGS, "rows": len(rows), "tip_close": prev,
            "pass": not failures, "failures": failures}


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    result = run(verbose=demo)
    print(json.dumps(result, sort_keys=True))
    sys.exit(0 if result["pass"] else 1)
