#!/usr/bin/env python3
"""Hardware-to-software seam emulator: USB-C connector (CC + SS pairs + D+/D-) -> JuniorDeck port.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md "Deck sheet" (Deck controls and Gaia helper
  separate; package ports JuniorLLM, AGI_SDK, JuniorOSai, JuniorOS, web3node,
  obsidian; vault not synced; device not opened), on the hardware class row
  from JuniorHome docs/HW_CLASS.md: USB-C (SS + D+/D-) -> JuniorDeck port.
  The seam decides, from the connector layer alone, whether the Deck sheet's
  port may be marked ready. "device not opened" stays true. Not ROADMAP T13
  (overnight owns it).

What it emulates
  In-memory fixtures of what a JuniorDeck acting as a USB-C sink (Rd 5.1 kOhm
  on each CC pin) would read: the voltage on CC1 and CC2, whether a SuperSpeed
  pair trained, whether the D+/D- pair enumerated, and the Deck's watts claim.
  1. Each CC voltage is classed with the Type-C sink windows:
       < 0.20 V open, 0.25-0.61 V Rp default USB (0.5 A / 0.9 A),
       0.70-1.16 V Rp 1.5 A, 1.31-2.04 V Rp 3.0 A, anything else out_of_band.
  2. Exactly one Rp pin -> attached, orientation normal (CC1) or flipped (CC2).
     Both open -> detached. Two Rp pins or any out_of_band pin -> reject
     (debug-accessory source or a bad cable; operator only).
  3. Data lane: SS trained -> usb3_ss on TX1/RX1 (normal) or TX2/RX2 (flipped);
     else D+/D- -> usb2; else none.
  4. Power: 5 V x advertised amps is the budget; power_ok if the claim fits.
  5. Deck port: port_ready = attached and a data lane and power_ok. The Deck
     sheet package ports are listed with opened False and synced False; the
     deck bind is the string 127.0.0.1:8766 only.
  One row per fixture is chained with close rule sha256(prev || pack || text)[:16]
  at rate 0.05, where pack is real 2-bit trit codes (0/1/2, four per byte,
  I2_S order) built from the per-pin, lane, and power states.

What it does not touch
  No /dev, /sys, sysfs typec class, configfs gadget, pyusb, or PD controller.
  It does not import or re-run web3node/deck_usb.py, scripts/deck_*_prod.py,
  ports/deck_sheet_port_map.py, or any other team file, and it does not cover
  the USB sound/HID class seam, SBC, SFF host, or spool (earlier 06:00 seams).
  No socket to the Deck app, bitnetd, i2sd, or any device: binds are strings.
  Nothing under ~/.juniorhome: rows go to a tempfile removed after the run.
  Standard library only. Flags: mint, socket, layer_one, fetch, ue5_launch False.

How to run
  python3 scripts/hwsw_usbc_cc_deck_port_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_usbc_cc_deck_port_seam.py demo     # emulated Deck port rows
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

RATE = 0.05
DECK_BIND = "127.0.0.1:8766"
BITNETD_BIND = "127.0.0.1:8765"
I2SD_BIND = "127.0.0.1:8767"
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16
VBUS_V = 5.0
PACKAGE_PORTS = ("JuniorLLM", "AGI_SDK", "JuniorOSai", "JuniorOS", "web3node", "obsidian")

# Type-C sink CC voltage windows (volts at the sink's Rd) -> (state, amps).
CC_WINDOWS = (
    (0.00, 0.20, "open", 0.0),
    (0.25, 0.61, "rp_default", 0.5),
    (0.70, 1.16, "rp_1a5", 1.5),
    (1.31, 2.04, "rp_3a0", 3.0),
)

FIXTURES = (
    {"name": "normal_3a0_ss", "cc1_v": 1.68, "cc2_v": 0.02, "ss": True, "dpdm": True, "claim_w": 12.0},
    {"name": "flipped_1a5_usb2", "cc1_v": 0.05, "cc2_v": 0.92, "ss": False, "dpdm": True, "claim_w": 6.0},
    {"name": "default_over_budget", "cc1_v": 0.41, "cc2_v": 0.00, "ss": True, "dpdm": True, "claim_w": 4.5},
    {"name": "detached", "cc1_v": 0.01, "cc2_v": 0.03, "ss": False, "dpdm": False, "claim_w": 0.0},
    {"name": "dual_rp_debug", "cc1_v": 0.95, "cc2_v": 0.95, "ss": False, "dpdm": True, "claim_w": 2.0},
    {"name": "out_of_band", "cc1_v": 2.60, "cc2_v": 0.00, "ss": True, "dpdm": True, "claim_w": 2.0},
    {"name": "attached_no_data", "cc1_v": 0.00, "cc2_v": 1.50, "ss": False, "dpdm": False, "claim_w": 3.0},
)


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def class_cc(volts: float) -> tuple[str, float]:
    for lo, hi, state, amps in CC_WINDOWS:
        if lo <= volts <= hi:
            return state, amps
    return "out_of_band", 0.0


def attach(cc1_v: float, cc2_v: float) -> dict:
    s1, a1 = class_cc(cc1_v)
    s2, a2 = class_cc(cc2_v)
    rp1, rp2 = s1.startswith("rp_"), s2.startswith("rp_")
    if "out_of_band" in (s1, s2):
        state, orient, amps, why = "reject", None, 0.0, "cc_out_of_band"
    elif rp1 and rp2:
        state, orient, amps, why = "reject", None, 0.0, "dual_rp"
    elif rp1:
        state, orient, amps, why = "attached", "normal", a1, ""
    elif rp2:
        state, orient, amps, why = "attached", "flipped", a2, ""
    else:
        state, orient, amps, why = "detached", None, 0.0, "no_rp"
    return {"cc1": s1, "cc2": s2, "state": state, "orientation": orient,
            "amps": amps, "reject": why}


def data_lane(att: dict, ss: bool, dpdm: bool) -> dict:
    if att["state"] != "attached":
        return {"lane": "none", "pairs": None}
    if ss:
        pairs = "TX1/RX1" if att["orientation"] == "normal" else "TX2/RX2"
        return {"lane": "usb3_ss", "pairs": pairs}
    if dpdm:
        return {"lane": "usb2", "pairs": "D+/D-"}
    return {"lane": "none", "pairs": None}


def power(att: dict, claim_w: float) -> dict:
    budget = round(VBUS_V * att["amps"], 3)
    return {"budget_w": budget, "claim_w": claim_w,
            "power_ok": att["state"] == "attached" and claim_w <= budget}


def deck_port(att: dict, lane: dict, pwr: dict) -> dict:
    ready = att["state"] == "attached" and lane["lane"] != "none" and pwr["power_ok"]
    return {
        "bind": DECK_BIND,
        "port_ready": ready,
        "device_opened": False,
        "vault_synced": False,
        "identity": {"deck_controls": "Deck controls", "gaia_helper": "Gaia helper",
                     "separate": True},
        "package_ports": [{"package": p, "opened": False, "synced": False}
                          for p in PACKAGE_PORTS],
        "live": False,
    }


def _trit(code_map: dict, key: str) -> int:
    return code_map[key]


def pack_trits(att: dict, lane: dict, pwr: dict) -> bytes:
    pin = {"open": 0, "rp_default": 1, "rp_1a5": 1, "rp_3a0": 1, "out_of_band": 2}
    codes = [
        _trit(pin, att["cc1"]),
        _trit(pin, att["cc2"]),
        {"none": 0, "usb2": 1, "usb3_ss": 2}[lane["lane"]],
        2 if pwr["power_ok"] else (1 if pwr["budget_w"] > 0 else 0),
        {"detached": 0, "attached": 1, "reject": 2}[att["state"]],
    ]
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


def build_row(prev: str, fx: dict) -> dict:
    att = attach(fx["cc1_v"], fx["cc2_v"])
    lane = data_lane(att, fx["ss"], fx["dpdm"])
    pwr = power(att, fx["claim_w"])
    port = deck_port(att, lane, pwr)
    pack = pack_trits(att, lane, pwr)
    text = json.dumps({"fixture": fx["name"], "attach": att, "lane": lane,
                       "power": pwr, "port_ready": port["port_ready"]}, sort_keys=True)
    return {
        "seam": "usb_c->juniordeck_port",
        "fixture": fx["name"],
        "attach": att,
        "lane": lane,
        "power": pwr,
        "deck_port": port,
        "binds": {"deck": DECK_BIND, "bitnetd": BITNETD_BIND, "i2sd": I2SD_BIND},
        "pack_hex": pack.hex(),
        "text": text,
        "rate": RATE,
        "prev": prev,
        "close": close(prev, pack, text),
        "fire": False,
        **FLAGS,
    }


def run(fixtures=FIXTURES) -> list[dict]:
    rows, prev = [], GENESIS
    for fx in fixtures:
        row = build_row(prev, fx)
        rows.append(row)
        prev = row["close"]
    return rows


def selftest() -> dict:
    rows = run()
    by = {r["fixture"]: r for r in rows}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "usbc_deck_rows.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        with open(path, encoding="utf-8") as fh:
            back = [json.loads(line) for line in fh]
        assert back == rows

    n = by["normal_3a0_ss"]
    assert n["attach"]["orientation"] == "normal" and n["attach"]["cc1"] == "rp_3a0"
    assert n["lane"] == {"lane": "usb3_ss", "pairs": "TX1/RX1"}
    assert n["power"]["budget_w"] == 15.0 and n["deck_port"]["port_ready"] is True
    f = by["flipped_1a5_usb2"]
    assert f["attach"]["orientation"] == "flipped" and f["lane"]["lane"] == "usb2"
    assert f["power"]["budget_w"] == 7.5 and f["deck_port"]["port_ready"] is True
    o = by["default_over_budget"]
    assert o["power"]["power_ok"] is False and o["deck_port"]["port_ready"] is False
    assert by["detached"]["attach"]["state"] == "detached"
    assert by["dual_rp_debug"]["attach"]["reject"] == "dual_rp"
    assert by["out_of_band"]["attach"]["reject"] == "cc_out_of_band"
    assert by["out_of_band"]["lane"]["lane"] == "none"
    assert by["attached_no_data"]["deck_port"]["port_ready"] is False
    # Window edges and gaps between windows.
    assert class_cc(0.20)[0] == "open" and class_cc(0.22)[0] == "out_of_band"
    assert class_cc(1.16)[0] == "rp_1a5" and class_cc(1.20)[0] == "out_of_band"

    prev = GENESIS
    for r in rows:
        assert r["prev"] == prev
        pack = bytes.fromhex(r["pack_hex"])
        assert len(pack) == 2 and all(c in (0, 1, 2) for c in unpack_trits(pack, 8))
        assert r["close"] == close(r["prev"], pack, r["text"])
        assert r["close"] != close(r["prev"], b"not-trit", r["text"])
        assert r["rate"] == RATE and r["fire"] is False
        assert not any(r[k] for k in FLAGS)
        port = r["deck_port"]
        assert port["device_opened"] is False and port["vault_synced"] is False
        assert port["live"] is False and port["bind"] == DECK_BIND
        assert [p["package"] for p in port["package_ports"]] == list(PACKAGE_PORTS)
        assert not any(p["opened"] or p["synced"] for p in port["package_ports"])
        for b in r["binds"].values():
            assert b.startswith("127.0.0.1:") and "0.0.0.0" not in b
        prev = r["close"]
    return {"selftest": "pass", "rows": len(rows),
            "ready": sorted(k for k, r in by.items() if r["deck_port"]["port_ready"]),
            "tip": rows[-1]["close"], "rate": RATE, **FLAGS}


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "demo":
        print(json.dumps(run(), indent=2))
    else:
        print(json.dumps(selftest(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
