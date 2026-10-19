#!/usr/bin/env python3
"""Hardware-to-software seam emulator: tun class (/sys/class/net/tun*) -> JuniorOS overlay loopback binds.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md "JuniorOS overlay" (bitnetd.service on
  127.0.0.1:8765, juniorctl on PATH, i2sd userspace on 127.0.0.1:8767, Deck on
  127.0.0.1:8766), on the hardware class row from JuniorHome docs/HW_CLASS.md:
  tun -> `/sys/class/net/tun*` -> operator probe only. The seam answers one
  question: when a tun/tap interface is present on the host, may the overlay
  still mark its services ready? Only if every overlay bind stays on
  127.0.0.1. The overlay never brings a tun up or down and never routes over
  it; tun handling stays an operator probe. Not ROADMAP T13 (overnight owns it).

What it emulates
  In-memory fixtures shaped like the netdev attributes Linux exposes under
  /sys/class/net/<if>/ (type, flags, tun_flags, operstate, mtu, owner) plus an
  emulated listener table of the overlay's bind strings.
  1. Interface class from the `type` attribute (ARPHRD): 772 loopback,
     65534 tun (ARPHRD_NONE), 1 ether; an ether netdev with a tun_flags
     attribute is a tap.
  2. tun_flags decode: IFF_TUN 0x0001, IFF_TAP 0x0002, IFF_MULTI_QUEUE 0x0100,
     IFF_NO_PI 0x1000. A tun whose tun_flags says TAP (or the reverse) is a
     mismatch and is rejected.
  3. Link state from `flags` (IFF_UP 0x1, IFF_POINTOPOINT 0x10, IFF_RUNNING
     0x40) and `operstate`: down, up_no_carrier, or up.
  4. Bind check: each overlay bind string must parse as host:port with host
     exactly 127.0.0.1 and port in the overlay set {8765, 8766, 8767}.
     0.0.0.0, ::, a tun-side address, or an unknown port fails.
  5. Host verdict: loopback present and up is required. Any up tun/tap plus
     any non-loopback bind -> reject (exposure over the tunnel). Up tun/tap
     with all binds loopback -> ready, tun marked operator_probe_only. No
     tun -> ready if binds are loopback.
  One row per host fixture is chained with close rule
  sha256(prev || pack || text)[:16] at rate 0.05, where pack is real 2-bit trit
  codes (0/1/2, four per byte, I2_S order, low bits first) built from each
  interface's link state and each bind's result.

What it does not touch
  No /sys, /dev/net/tun, ioctl(TUNSETIFF), ip/iproute2, netlink, nftables, or
  route table. No socket is opened: binds are strings and are never bound.
  It does not import or re-run bitnetd, i2sd, juniorctl, scripts/advance_*,
  scripts/tuner_*, ports/*, or any other team file, and it does not cover the
  earlier 06:00 seams (spool, USB sound/HID, SBC, SFF host, USB-C).
  Nothing under ~/.juniorhome: rows go to a tempfile removed after the run.
  Standard library only. Flags: mint, socket, layer_one, fetch, ue5_launch False.

How to run
  python3 scripts/hwsw_tun_loopback_overlay_bind_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_tun_loopback_overlay_bind_seam.py demo     # emulated host rows
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
LOOPBACK = "127.0.0.1"
OVERLAY_PORTS = {8765: "bitnetd", 8766: "deck", 8767: "i2sd"}
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16

ARPHRD_ETHER = 1
ARPHRD_LOOPBACK = 772
ARPHRD_NONE = 65534

IFF_UP = 0x1
IFF_LOOPBACK = 0x8
IFF_POINTOPOINT = 0x10
IFF_RUNNING = 0x40

TUN_IFF_TUN = 0x0001
TUN_IFF_TAP = 0x0002
TUN_IFF_MULTI_QUEUE = 0x0100
TUN_IFF_NO_PI = 0x1000


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def _hex(v) -> int:
    if isinstance(v, int):
        return v
    return int(str(v), 16) if str(v).lower().startswith("0x") else int(str(v))


def classify(name: str, attrs: dict) -> dict:
    """Class one netdev from sysfs-shaped attributes (strings or ints)."""
    typ = _hex(attrs.get("type", 0))
    flags = _hex(attrs.get("flags", 0))
    tf_raw = attrs.get("tun_flags")
    tf = _hex(tf_raw) if tf_raw is not None else None
    why = ""
    if typ == ARPHRD_LOOPBACK or flags & IFF_LOOPBACK:
        cls = "loopback"
    elif typ == ARPHRD_NONE and tf is not None:
        cls = "tun"
        if not tf & TUN_IFF_TUN or tf & TUN_IFF_TAP:
            cls, why = "reject", "tun_flags_mismatch"
    elif typ == ARPHRD_ETHER and tf is not None:
        cls = "tap"
        if not tf & TUN_IFF_TAP or tf & TUN_IFF_TUN:
            cls, why = "reject", "tun_flags_mismatch"
    elif typ == ARPHRD_ETHER:
        cls = "ether"
    else:
        cls, why = "other", "unknown_type"

    oper = str(attrs.get("operstate", "unknown")).strip()
    if not flags & IFF_UP or oper == "down":
        link = "down"
    elif flags & IFF_RUNNING or oper in ("up", "unknown") and cls == "loopback":
        link = "up"
    else:
        link = "up_no_carrier"

    out = {"if": name, "class": cls, "link": link, "why": why,
           "mtu": int(attrs.get("mtu", 0))}
    if tf is not None:
        out["tun_flags"] = {
            "tun": bool(tf & TUN_IFF_TUN),
            "tap": bool(tf & TUN_IFF_TAP),
            "multi_queue": bool(tf & TUN_IFF_MULTI_QUEUE),
            "no_pi": bool(tf & TUN_IFF_NO_PI),
        }
        out["point_to_point"] = bool(flags & IFF_POINTOPOINT)
        out["operator_probe_only"] = True
    return out


def check_bind(bind: str) -> dict:
    host, sep, port_s = bind.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        port = int(port_s) if sep else -1
    except ValueError:
        port = -1
    if host != LOOPBACK:
        state = "exposed"
    elif port not in OVERLAY_PORTS:
        state = "unknown_port"
    else:
        state = "loopback"
    return {"bind": bind, "host": host, "port": port,
            "service": OVERLAY_PORTS.get(port), "state": state}


def verdict(ifaces: list[dict], binds: list[dict]) -> dict:
    lo_up = any(i["class"] == "loopback" and i["link"] == "up" for i in ifaces)
    tunnels = [i for i in ifaces if i["class"] in ("tun", "tap")]
    tun_up = any(i["link"] == "up" for i in tunnels)
    bad = [b for b in binds if b["state"] != "loopback"]
    rejected = [i for i in ifaces if i["class"] == "reject"]
    if not lo_up:
        state, why = "reject", "no_loopback"
    elif rejected:
        state, why = "reject", rejected[0]["why"]
    elif bad and tun_up:
        state, why = "reject", "exposed_over_tunnel"
    elif bad:
        state, why = "reject", "non_loopback_bind"
    else:
        state, why = "ready", ""
    return {"state": state, "why": why, "tun_present": bool(tunnels),
            "tun_up": tun_up, "tun_role": "operator_probe_only" if tunnels else None,
            "overlay_routes_tun": False, "overlay_touches_tun": False}


def pack_trits(ifaces: list[dict], binds: list[dict], v: dict) -> bytes:
    link = {"down": 0, "up_no_carrier": 1, "up": 2}
    bind = {"loopback": 0, "unknown_port": 1, "exposed": 2}
    codes = [link[i["link"]] if i["class"] != "reject" else 2 for i in ifaces]
    codes += [bind[b["state"]] for b in binds]
    codes.append({"ready": 0, "reject": 2}[v["state"]] if not v["tun_present"]
                 else {"ready": 1, "reject": 2}[v["state"]])
    while len(codes) % 4:
        codes.append(0)
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            assert c in (0, 1, 2)
            b |= c << (2 * j)
        out.append(b)
    return bytes(out)


def unpack_trits(pack: bytes, n: int) -> list[int]:
    return [(pack[k // 4] >> (2 * (k % 4))) & 3 for k in range(n)]


LO = {"type": "772", "flags": "0x9", "operstate": "unknown", "mtu": "65536"}
GOOD_BINDS = ["127.0.0.1:8765", "127.0.0.1:8766", "127.0.0.1:8767"]

FIXTURES = [
    {"host": "sff_no_tun", "ifaces": {"lo": LO,
        "eth0": {"type": "1", "flags": "0x1043", "operstate": "up", "mtu": "1500"}},
     "binds": GOOD_BINDS, "want": "ready"},
    {"host": "sbc_tun_up_loopback_binds", "ifaces": {"lo": LO,
        "tun0": {"type": "65534", "flags": "0x10d1", "operstate": "unknown",
                 "mtu": "1420", "tun_flags": "0x1001", "owner": "-1"}},
     "binds": GOOD_BINDS, "want": "ready"},
    {"host": "sbc_tun_up_wildcard_bitnetd", "ifaces": {"lo": LO,
        "tun0": {"type": "65534", "flags": "0x10d1", "operstate": "unknown",
                 "mtu": "1420", "tun_flags": "0x1001"}},
     "binds": ["0.0.0.0:8765", "127.0.0.1:8766", "127.0.0.1:8767"],
     "want": "reject", "why": "exposed_over_tunnel"},
    {"host": "sff_tap_down_ipv6_any", "ifaces": {"lo": LO,
        "tap0": {"type": "1", "flags": "0x1002", "operstate": "down",
                 "mtu": "1500", "tun_flags": "0x1102"}},
     "binds": ["127.0.0.1:8765", "[::]:8766", "127.0.0.1:8767"],
     "want": "reject", "why": "non_loopback_bind"},
    {"host": "tun_flags_say_tap", "ifaces": {"lo": LO,
        "tun1": {"type": "65534", "flags": "0x10d1", "operstate": "unknown",
                 "mtu": "1500", "tun_flags": "0x1002"}},
     "binds": GOOD_BINDS, "want": "reject", "why": "tun_flags_mismatch"},
    {"host": "loopback_down", "ifaces": {
        "lo": {"type": "772", "flags": "0x8", "operstate": "down", "mtu": "65536"}},
     "binds": GOOD_BINDS, "want": "reject", "why": "no_loopback"},
    {"host": "tun_no_carrier_stray_port", "ifaces": {"lo": LO,
        "tun0": {"type": "65534", "flags": "0x1091", "operstate": "down",
                 "mtu": "1420", "tun_flags": "0x1001"}},
     "binds": ["127.0.0.1:8765", "127.0.0.1:8766", "127.0.0.1:9999"],
     "want": "reject", "why": "non_loopback_bind"},
]


def build_row(prev: str, fx: dict) -> dict:
    ifaces = [classify(n, a) for n, a in sorted(fx["ifaces"].items())]
    binds = [check_bind(b) for b in fx["binds"]]
    v = verdict(ifaces, binds)
    pack = pack_trits(ifaces, binds, v)
    body = {"host": fx["host"], "ifaces": ifaces, "binds": binds,
            "verdict": v, "rate": RATE, "flags": FLAGS}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return {"body": body, "pack": pack.hex(), "prev": prev,
            "close": close(prev, pack, text), "n_trits": len(ifaces) + len(binds) + 1}


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
        v = r["body"]["verdict"]
        ok = v["state"] == fx["want"] and v["why"] == fx.get("why", "")
        pack = bytes.fromhex(r["pack"])
        codes = unpack_trits(pack, r["n_trits"])
        ok = ok and all(c in (0, 1, 2) for c in codes)
        ok = ok and v["overlay_routes_tun"] is False and v["overlay_touches_tun"] is False
        results.append({"host": fx["host"], "state": v["state"], "why": v["why"], "ok": ok})
    # chain re-check: recompute every close from its stored prev/pack/body
    prev, chain_ok = GENESIS, True
    for r in rows:
        text = json.dumps(r["body"], sort_keys=True, separators=(",", ":"))
        chain_ok &= r["prev"] == prev and close(prev, bytes.fromhex(r["pack"]), text) == r["close"]
        prev = r["close"]
    # tempfile round trip, removed after
    fd, path = tempfile.mkstemp(prefix="hwsw_tun_", suffix=".jsonl")
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
    return {"seam": "tun_class->junioros_overlay_loopback_binds", "rate": RATE,
            "rows": results, "chain_ok": chain_ok, "disk_ok": disk_ok,
            "flags": FLAGS, "tip_close": rows[-1]["close"], "pass": passed}


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "demo":
        for r in run_rows():
            print(json.dumps({"host": r["body"]["host"], "verdict": r["body"]["verdict"],
                              "pack": r["pack"], "close": r["close"]}, sort_keys=True))
        return 0
    res = self_test()
    print(json.dumps(res, indent=2, sort_keys=True))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
