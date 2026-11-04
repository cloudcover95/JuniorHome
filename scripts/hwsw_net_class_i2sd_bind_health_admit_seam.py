#!/usr/bin/env python3
"""Hardware-to-software seam emulator: /sys/class/net + /proc/net/dev -> i2sd bind/health admit.

Roadmap item
  JuniorHome docs/ROADMAP.md item 3, "i2sd userspace -- 127.0.0.1:8767. No
  docker.sock." The seam decides, from the net class layer alone, whether the
  i2sd userspace daemon may admit listen/health on a box. Hardware classes:
  SBC (arm64, JuniorOS T4, mtu<=1500) and SFF host (JuniorHome T0, jumbo
  allowed). Not ROADMAP T13 (overnight owns it). Not item 4 (no Asahi/MLX).
  Not item 5 (ue5_launch stays False). Not a re-run of the tun/loopback
  overlay bind seam, power_supply admit, thermal/cpufreq, or DRM EDID seams:
  this is operstate/carrier/mtu/speed/address + optional /proc/net/dev
  counters for i2sd listen health only.

What it emulates
  In-memory fixtures shaped like /sys/class/net/<if>/{operstate,carrier,mtu,
  speed,address} plus optional /proc/net/dev counter lines {rx_bytes,
  tx_bytes, rx_errs, tx_errs}. Rules, in order:
  1. Bind host:port must be exactly 127.0.0.1:8767; any other host
     (including 0.0.0.0) -> reject (decision "reject", flag bind_reject).
  2. Class must be "sbc" or "sff"; otherwise -> "hold".
  3. Parse operstate (up/down/unknown/...), carrier (0/1), mtu (int),
     optional speed (Mb/s or -1), address (MAC string). Unparsed operstate,
     carrier, or mtu -> "hold" (flag unparsed).
  4. operstate must be "up" AND carrier==1 for admit; carrier down (or
     operstate down with carrier 0) -> "soft" (boot-relative retry, flag
     carrier_down). operstate unknown with carrier present -> "hold".
  5. mtu must fall in [1280, 9000]. Outside that band -> "hold"
     (flag mtu_oob). SBC class additionally requires mtu<=1500; above
     that on SBC -> "hold" (flag sbc_jumbo). SFF may use jumbo up to 9000.
  6. Optional counters: if present, rx_errs or tx_errs > 0 with zero
     successful bytes -> "soft" (flag link_errors); unparsed counters are
     ignored (do not hold).
  7. Admit only when bind ok, operstate=up, carrier=1, mtu in range, and
     class MTU rule passes.
  Each row packs decision and check states as real 2-bit trit codes
  (0/1/2, four per byte, I2_S order: first code in the low bits), and rows
  are chained with close rule sha256(prev || pack || text)[:16] at rate 0.05.

What it does not touch
  No real /sys or /proc reads, no ioctl, no ip/iproute2, no netlink, no
  nftables, no real NIC opens, no sockets (binds are strings; nothing
  listens or connects), no device opened, no model pull, no GGUF, no
  network fetch, no subprocess, no docker.sock. It does not import or
  re-run any team file or any earlier hwsw_* seam (tun/loopback, power,
  thermal, DRM EDID, backlight, USB-C, and so on). Rows go to a tempfile
  removed after the run; nothing under ~/.juniorhome. Flags: mint, socket,
  layer_one, fetch, ue5_launch all False. JuniorSolana and the profile
  repo are untouched.

How to run
  python3 scripts/hwsw_net_class_i2sd_bind_health_admit_seam.py        # self-test, JSON, exit 0 on pass
  python3 scripts/hwsw_net_class_i2sd_bind_health_admit_seam.py demo   # one line per fixture row
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
BIND_OK = "127.0.0.1:8767"
MTU_MIN = 1280
MTU_MAX = 9000
MTU_SBC_CAP = 1500
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
DECISION_CODE = {"reject": 0, "hold": 0, "soft": 1, "admit": 2}


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


def num(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def evaluate(fx: dict) -> dict:
    cls = fx.get("cls")
    bind = fx.get("bind", "")
    net = fx.get("net", {})
    counters = fx.get("counters")  # optional dict

    bind_ok = bind == BIND_OK
    hold_reason = ""
    decision = "hold"

    if not bind_ok:
        return _row(fx, "reject", "bind_reject", bind_ok=False, link_ok=False,
                    mtu_ok=False, carrier=None, mtu=None, operstate=None)

    if cls not in ("sbc", "sff"):
        return _row(fx, "hold", "bad_class", bind_ok=True, link_ok=False,
                    mtu_ok=False, carrier=None, mtu=None, operstate=None)

    operstate = str(net.get("operstate", "")).strip().lower()
    carrier_v = num(net.get("carrier", ""))
    mtu_v = num(net.get("mtu", ""))
    # speed may be -1 (unknown) — optional, not a hold by itself
    _speed = num(net.get("speed", ""))
    _addr = str(net.get("address", "")).strip()

    if operstate not in ("up", "down", "unknown", "dormant", "lowerlayerdown"):
        return _row(fx, "hold", "unparsed", bind_ok=True, link_ok=False,
                    mtu_ok=False, carrier=carrier_v, mtu=mtu_v, operstate=operstate or None)
    if carrier_v is None or carrier_v not in (0, 1):
        return _row(fx, "hold", "unparsed", bind_ok=True, link_ok=False,
                    mtu_ok=False, carrier=carrier_v, mtu=mtu_v, operstate=operstate)
    if mtu_v is None:
        return _row(fx, "hold", "unparsed", bind_ok=True, link_ok=False,
                    mtu_ok=False, carrier=carrier_v, mtu=None, operstate=operstate)

    mtu_ok = MTU_MIN <= mtu_v <= MTU_MAX
    if not mtu_ok:
        return _row(fx, "hold", "mtu_oob", bind_ok=True, link_ok=False,
                    mtu_ok=False, carrier=carrier_v, mtu=mtu_v, operstate=operstate)
    if cls == "sbc" and mtu_v > MTU_SBC_CAP:
        return _row(fx, "hold", "sbc_jumbo", bind_ok=True, link_ok=False,
                    mtu_ok=False, carrier=carrier_v, mtu=mtu_v, operstate=operstate)

    if operstate == "unknown":
        return _row(fx, "hold", "unparsed", bind_ok=True, link_ok=False,
                    mtu_ok=True, carrier=carrier_v, mtu=mtu_v, operstate=operstate)

    link_up = operstate == "up" and carrier_v == 1
    if not link_up:
        # carrier down / operstate down -> soft (boot-relative retry)
        return _row(fx, "soft", "carrier_down", bind_ok=True, link_ok=False,
                    mtu_ok=True, carrier=carrier_v, mtu=mtu_v, operstate=operstate)

    # optional counters: errors with no success -> soft
    if isinstance(counters, dict):
        rx_b = num(counters.get("rx_bytes", "0"))
        tx_b = num(counters.get("tx_bytes", "0"))
        rx_e = num(counters.get("rx_errs", "0"))
        tx_e = num(counters.get("tx_errs", "0"))
        # only act when all four parse; otherwise ignore
        if None not in (rx_b, tx_b, rx_e, tx_e):
            if (rx_e > 0 or tx_e > 0) and (rx_b + tx_b) == 0:
                return _row(fx, "soft", "link_errors", bind_ok=True, link_ok=True,
                            mtu_ok=True, carrier=carrier_v, mtu=mtu_v, operstate=operstate)

    return _row(fx, "admit", None, bind_ok=True, link_ok=True,
                mtu_ok=True, carrier=carrier_v, mtu=mtu_v, operstate=operstate)


def _row(fx, decision, hold_reason, bind_ok, link_ok, mtu_ok, carrier, mtu, operstate):
    trits = [
        DECISION_CODE[decision],
        2 if bind_ok else 0,
        2 if link_ok else (1 if decision == "soft" else 0),
        2 if mtu_ok else 0,
        2 if fx.get("cls") == "sff" else (1 if fx.get("cls") == "sbc" else 0),
        2 if decision == "admit" else (1 if decision == "soft" else 0),
    ]
    return {
        "name": fx["name"],
        "cls": fx.get("cls"),
        "bind_ok": bind_ok,
        "link_ok": link_ok,
        "mtu_ok": mtu_ok,
        "carrier": carrier,
        "mtu": mtu,
        "operstate": operstate,
        "decision": decision,
        "hold": hold_reason,
        "admit": decision == "admit",
        "trits": trits,
    }


FIXTURES = [
    {
        "name": "sff_eth_up_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "1", "mtu": "1500",
                "speed": "1000", "address": "aa:bb:cc:dd:ee:01"},
        "counters": {"rx_bytes": "4096", "tx_bytes": "2048", "rx_errs": "0", "tx_errs": "0"},
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_lo_up_admit",
        "cls": "sbc",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "1", "mtu": "1500",
                "speed": "-1", "address": "00:00:00:00:00:00"},
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sff_jumbo_admit",
        "cls": "sff",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "1", "mtu": "9000",
                "speed": "10000", "address": "aa:bb:cc:dd:ee:02"},
        "want": {"decision": "admit", "admit": True, "hold": None},
    },
    {
        "name": "sbc_jumbo_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "1", "mtu": "9000",
                "speed": "1000", "address": "aa:bb:cc:dd:ee:03"},
        "want": {"decision": "hold", "admit": False, "hold": "sbc_jumbo"},
    },
    {
        "name": "sff_carrier_down_soft",
        "cls": "sff",
        "bind": BIND_OK,
        "net": {"operstate": "down", "carrier": "0", "mtu": "1500",
                "speed": "1000", "address": "aa:bb:cc:dd:ee:04"},
        "want": {"decision": "soft", "admit": False, "hold": "carrier_down"},
    },
    {
        "name": "sbc_operstate_unknown_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "net": {"operstate": "unknown", "carrier": "1", "mtu": "1500",
                "speed": "-1", "address": "00:00:00:00:00:00"},
        "want": {"decision": "hold", "admit": False, "hold": "unparsed"},
    },
    {
        "name": "sff_mtu_oob_hold",
        "cls": "sff",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "1", "mtu": "512",
                "speed": "100", "address": "aa:bb:cc:dd:ee:05"},
        "want": {"decision": "hold", "admit": False, "hold": "mtu_oob"},
    },
    {
        "name": "sff_wildcard_bind_reject",
        "cls": "sff",
        "bind": "0.0.0.0:8767",
        "net": {"operstate": "up", "carrier": "1", "mtu": "1500",
                "speed": "1000", "address": "aa:bb:cc:dd:ee:06"},
        "want": {"decision": "reject", "admit": False, "hold": "bind_reject"},
    },
    {
        "name": "sbc_carrier_unparsed_hold",
        "cls": "sbc",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "n/a", "mtu": "1500",
                "speed": "-1", "address": "00:00:00:00:00:00"},
        "want": {"decision": "hold", "admit": False, "hold": "unparsed"},
    },
    {
        "name": "sff_link_errors_soft",
        "cls": "sff",
        "bind": BIND_OK,
        "net": {"operstate": "up", "carrier": "1", "mtu": "1500",
                "speed": "1000", "address": "aa:bb:cc:dd:ee:07"},
        "counters": {"rx_bytes": "0", "tx_bytes": "0", "rx_errs": "12", "tx_errs": "0"},
        "want": {"decision": "soft", "admit": False, "hold": "link_errors"},
    },
]


def run(verbose: bool = False) -> dict:
    prev = GENESIS
    rows = []
    fails = []
    fd, path = tempfile.mkstemp(prefix="hwsw_net_", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as fh:
            for fx in FIXTURES:
                r = evaluate(fx)
                text = json.dumps({k: v for k, v in r.items() if k != "trits"}, sort_keys=True)
                prev = close(prev, pack2(r["trits"]), text)
                r["close"] = prev
                fh.write(json.dumps(r, sort_keys=True) + "\n")
                ok = all(r.get(k) == v for k, v in fx["want"].items())
                if not ok:
                    fails.append({"name": fx["name"], "got": {k: r.get(k) for k in fx["want"]},
                                  "want": fx["want"]})
                rows.append(r)
                if verbose:
                    print(f"{r['name']:<28} decision={r['decision']!s:<7} "
                          f"hold={r['hold']!s:<14} mtu={r['mtu']!s:<5} "
                          f"carrier={r['carrier']!s:<3} close={prev}")
    finally:
        os.remove(path)
    return {
        "seam": "net_class->i2sd_bind_health",
        "roadmap": "ROADMAP #3 i2sd userspace",
        "rate": RATE,
        "bind": BIND_OK,
        "flags": FLAGS,
        "rows": len(rows),
        "fail": [f["name"] if isinstance(f, dict) else f for f in fails],
        "pass": not fails,
        "tip_close": prev,
    }


if __name__ == "__main__":
    demo = len(sys.argv) > 1 and sys.argv[1] == "demo"
    out = run(verbose=demo)
    if not demo:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({k: out[k] for k in ("pass", "tip_close", "fail", "rows")}, indent=2))
    sys.exit(0 if out["pass"] else 1)
