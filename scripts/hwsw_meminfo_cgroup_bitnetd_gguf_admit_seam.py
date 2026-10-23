#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /proc/meminfo + cgroup v2 memory.max -> bitnetd GGUF admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 2, "GGUF — operator sets JUNIOR_GGUF. Bots
  do not download.", at the seam where an operator-set model file meets the
  RAM of the hardware class running bitnetd.service on 127.0.0.1:8765
  (SBC arm64 <=12 W -> JuniorOS T4, SFF host x86_64/arm64 -> JuniorHome T0).
  The seam answers one question: given what the board and the bitnetd cgroup
  say about memory, may bitnetd map the GGUF the operator pointed at, and with
  how large a KV context? Rule: memory pressure only lowers the context or
  holds; bitnetd never evicts a loaded model, never swaps the weights, and
  never fetches a smaller model on its own. Not ROADMAP T13 (overnight owns
  it). Not a re-derivation of any earlier seam file: no other team file is
  imported or read.

What it emulates
  In-memory fixtures shaped like kernel text:
    /proc/meminfo                 (MemTotal, MemAvailable, SwapTotal, kB)
    /sys/fs/cgroup/system.slice/bitnetd.service/memory.max   ("max" or bytes)
    /sys/fs/cgroup/system.slice/bitnetd.service/memory.current (bytes)
    /sys/fs/cgroup/system.slice/bitnetd.service/memory.pressure
      ("some avg10=.. avg60=.. avg300=.. total=..")
  plus JUNIOR_GGUF as a declared path and a declared stat size (the file is
  never opened or stat'ed), the declared parameter count, the hardware class
  and the bitnetd bind.
  1. Parse: meminfo kB lines; memory.max "max" means no cgroup limit. Any
     unparsable field fails safe to "hold".
  2. Budget: min(MemAvailable, memory.max - memory.current) minus a class
     reserve (SBC 256 MiB, SFF 1 GiB).
  3. Weights: the GGUF must be present (JUNIOR_GGUF set, path ends .gguf);
     its declared size must be within 10 % of the I2_S estimate
     params * 2 bit / 8 (a size far off means not an I2_S pack -> reject).
  4. KV: per-token KV bytes from the class profile; context is the largest
     power of two in [512, class max] that fits budget - weights. None fits
     -> "hold". memory.pressure some avg10 >= 10.0 halves the context
     (floor 512, else hold). SwapTotal > 0 on an SBC is noted, never used.
  5. Bind: bitnetd must be exactly 127.0.0.1:8765; anything else rejects.
  6. Exactness: a fixed ternary block (codes 0/1/2 = -1/0/+1, four per byte,
     low bits first) is packed, unpacked and dotted with int8 activations;
     the row passes only if it equals the unpacked scalar reference, so the
     I2_S byte estimate in step 3 is the real pack size.
  One row per fixture is chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05, where pack is the 2-bit
  trit weight block plus trit codes for the verdict and bind.

What it does not touch
  No real /proc or /sys reads, no cgroup writes, no systemd calls, no file
  opened at JUNIOR_GGUF, no model pull, no GGUF download, no mmap, no
  sockets (nothing binds or connects), no device opened, no network fetch,
  no subprocess. mint/socket/layer_one/fetch/ue5_launch are all false.
  JuniorSolana and the profile repo are untouched.

How to run
  python3 scripts/hwsw_meminfo_cgroup_bitnetd_gguf_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_meminfo_cgroup_bitnetd_gguf_admit_seam.py demo   # one line per fixture row
  Needs only Python 3.8+ stdlib.
"""
from __future__ import annotations

import hashlib
import json
import sys

RATE = 0.05
GENESIS = "0" * 16
BIND_OK = "127.0.0.1:8765"
MIB = 1024 * 1024
GIB = 1024 * MIB
RESERVE = {"sbc": 256 * MIB, "sff": 1 * GIB}
# per-token KV bytes and max context by class profile
KV_PER_TOKEN = {"sbc": 64 * 1024, "sff": 160 * 1024}
CTX_MAX = {"sbc": 4096, "sff": 16384}
CTX_MIN = 512
PRESSURE_HALVE = 10.0
SIZE_TOL = 0.10
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}

W_CODES = [(i * 5 + 1) % 3 for i in range(128)]
ACTS = [((i * 41) % 255) - 127 for i in range(128)]


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def pack2(codes: list) -> bytes:
    out = bytearray((len(codes) + 3) // 4)
    for i, c in enumerate(codes):
        out[i // 4] |= (c & 3) << (2 * (i % 4))
    return bytes(out)


def unpack2(buf: bytes, n: int) -> list:
    return [(buf[i // 4] >> (2 * (i % 4))) & 3 for i in range(n)]


def dot(codes: list, acts: list) -> int:
    return sum((c - 1) * a for c, a in zip(codes, acts))


def i2s_bytes(params: int) -> int:
    return (params * 2 + 7) // 8


def parse_meminfo(text: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if not parts or (len(parts) > 1 and parts[1] != "kB"):
            raise ValueError("bad meminfo line: " + line)
        out[k.strip()] = int(parts[0]) * 1024
    for need in ("MemTotal", "MemAvailable"):
        if need not in out:
            raise ValueError("missing " + need)
    return out


def parse_max(text: str):
    t = text.strip()
    return None if t == "max" else int(t)


def parse_pressure(text: str) -> float:
    for line in text.strip().splitlines():
        f = line.split()
        if f and f[0] == "some":
            kv = dict(p.split("=", 1) for p in f[1:])
            return float(kv["avg10"])
    raise ValueError("no some line")


def admit(fx: dict) -> dict:
    cls = fx["cls"]
    if fx["bind"] != BIND_OK:
        return {"verdict": "reject", "ctx": 0, "why": "bind_not_loopback_8765", "budget": 0}
    gguf = fx.get("junior_gguf") or ""
    if not gguf or not gguf.endswith(".gguf"):
        return {"verdict": "hold", "ctx": 0, "why": "junior_gguf_unset_operator_sets_it", "budget": 0}
    try:
        mi = parse_meminfo(fx["meminfo"])
        mmax = parse_max(fx["memory_max"])
        cur = int(fx["memory_current"].strip())
        avg10 = parse_pressure(fx["memory_pressure"])
    except (ValueError, KeyError):
        return {"verdict": "hold", "ctx": 0, "why": "unparsed_fail_safe", "budget": 0}
    avail = mi["MemAvailable"]
    if mmax is not None:
        avail = min(avail, mmax - cur)
    budget = avail - RESERVE[cls]
    est = i2s_bytes(fx["params"])
    size = fx["gguf_size"]
    if abs(size - est) > SIZE_TOL * est:
        return {"verdict": "reject", "ctx": 0, "why": "size_not_i2s_pack", "budget": budget}
    room = budget - size
    ctx = 0
    c = CTX_MAX[cls]
    while c >= CTX_MIN:
        if c * KV_PER_TOKEN[cls] <= room:
            ctx = c
            break
        c //= 2
    why = "fits"
    if ctx and avg10 >= PRESSURE_HALVE:
        ctx = ctx // 2 if ctx // 2 >= CTX_MIN else 0
        why = "pressure_halved"
    if not ctx:
        return {"verdict": "hold", "ctx": 0, "why": "no_ctx_fits" if why == "fits" else "pressure_hold",
                "budget": budget}
    note = "sbc_swap_seen_not_used" if cls == "sbc" and mi.get("SwapTotal", 0) > 0 else ""
    return {"verdict": "admit", "ctx": ctx, "why": why + ("+" + note if note else ""), "budget": budget}


def mi(total_mib: int, avail_mib: int, swap_mib: int = 0) -> str:
    return ("MemTotal:       %d kB\nMemAvailable:   %d kB\nSwapTotal:      %d kB\n"
            % (total_mib * 1024, avail_mib * 1024, swap_mib * 1024))


def pr(avg10: float) -> str:
    return "some avg10=%.2f avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0" % avg10


P2B = 2_400_000_000  # 2.4B params -> 600 MB I2_S
FIXTURES = [
    {"host": "sbc_4g_admit", "cls": "sbc", "bind": BIND_OK, "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": i2s_bytes(P2B), "meminfo": mi(4096, 2600), "memory_max": "max",
     "memory_current": "0", "memory_pressure": pr(0.5), "expect": ("admit", 4096)},
    {"host": "sbc_2g_small_ctx", "cls": "sbc", "bind": BIND_OK, "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": i2s_bytes(P2B), "meminfo": mi(2048, 1000, 512), "memory_max": "max",
     "memory_current": "0", "memory_pressure": pr(0.0), "expect": ("admit", 2048)},
    {"host": "sbc_cgroup_cap_hold", "cls": "sbc", "bind": BIND_OK, "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": i2s_bytes(P2B), "meminfo": mi(4096, 3000), "memory_max": str(840 * MIB),
     "memory_current": str(10 * MIB), "memory_pressure": pr(0.0), "expect": ("hold", 0)},
    {"host": "sff_pressure_halved", "cls": "sff", "bind": BIND_OK, "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": i2s_bytes(P2B), "meminfo": mi(16384, 9000), "memory_max": str(6 * GIB),
     "memory_current": str(100 * MIB), "memory_pressure": pr(22.5), "expect": ("admit", 8192)},
    {"host": "sff_f16_size_reject", "cls": "sff", "bind": BIND_OK, "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": P2B * 2, "meminfo": mi(32768, 30000), "memory_max": "max",
     "memory_current": "0", "memory_pressure": pr(0.0), "expect": ("reject", 0)},
    {"host": "sff_gguf_unset_hold", "cls": "sff", "bind": BIND_OK, "junior_gguf": "",
     "params": P2B, "gguf_size": 0, "meminfo": mi(16384, 9000), "memory_max": "max",
     "memory_current": "0", "memory_pressure": pr(0.0), "expect": ("hold", 0)},
    {"host": "sbc_unparsed_hold", "cls": "sbc", "bind": BIND_OK, "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": i2s_bytes(P2B), "meminfo": "MemTotal: lots\n", "memory_max": "max",
     "memory_current": "0", "memory_pressure": pr(0.0), "expect": ("hold", 0)},
    {"host": "sff_bad_bind", "cls": "sff", "bind": "0.0.0.0:8765", "junior_gguf": "/srv/models/op.gguf",
     "params": P2B, "gguf_size": i2s_bytes(P2B), "meminfo": mi(16384, 9000), "memory_max": "max",
     "memory_current": "0", "memory_pressure": pr(0.0), "expect": ("reject", 0)},
]

VERDICT_TRIT = {"admit": 2, "hold": 1, "reject": 0}


def run_rows() -> list:
    prev = GENESIS
    rows = []
    wpack = pack2(W_CODES)
    ref = dot(W_CODES, ACTS)
    got = dot(unpack2(wpack, len(W_CODES)), ACTS)
    for fx in FIXTURES:
        a = admit(fx)
        body = {"host": fx["host"], "cls": fx["cls"], "verdict": a["verdict"], "ctx": a["ctx"],
                "why": a["why"], "budget": a["budget"], "dot": got, "exact": got == ref,
                "pack_bytes": len(wpack), "rate": RATE}
        text = json.dumps(body, sort_keys=True)
        pack = wpack + pack2([VERDICT_TRIT[a["verdict"]], 2 if fx["bind"] == BIND_OK else 0])
        c = close(prev, pack, text)
        rows.append({"prev": prev, "body": body, "text": text, "pack": pack.hex(), "close": c,
                     "expect": fx["expect"]})
        prev = c
    return rows


def self_test() -> dict:
    rows = run_rows()
    results = []
    for r in rows:
        b = r["body"]
        ok = ((b["verdict"], b["ctx"]) == tuple(r["expect"]) and b["exact"]
              and b["ctx"] <= CTX_MAX[b["cls"]] and b["pack_bytes"] == i2s_bytes(len(W_CODES)))
        results.append({"host": b["host"], "verdict": b["verdict"], "ctx": b["ctx"], "ok": ok})
    chain_ok = True
    prev = GENESIS
    for r in rows:
        chain_ok &= (r["prev"] == prev and close(prev, bytes.fromhex(r["pack"]), r["text"]) == r["close"])
        prev = r["close"]
    # monotonic: shrinking MemAvailable on the same SBC never raises ctx
    mono_ok = True
    last = None
    for avail in range(4000, 300, -100):
        n = admit(dict(FIXTURES[0], meminfo=mi(4096, avail)))["ctx"]
        if last is not None and n > last:
            mono_ok = False
        last = n
    flags_ok = not any(FLAGS.values())
    passed = all(x["ok"] for x in results) and chain_ok and mono_ok and flags_ok
    return {"seam": "meminfo_cgroup_bitnetd_gguf_admit", "roadmap": "ROADMAP #2 GGUF (operator sets JUNIOR_GGUF)",
            "bind": BIND_OK, "rows": results, "chain_ok": chain_ok, "mono_ok": mono_ok,
            "flags": FLAGS, "rate": RATE, "tip_close": rows[-1]["close"], "pass": passed}


def main(argv: list) -> int:
    if len(argv) > 1 and argv[1] == "demo":
        for r in run_rows():
            print(r["close"], r["text"])
        return 0
    res = self_test()
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
