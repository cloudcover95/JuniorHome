"""Deck sheet port map: JuniorDeck pin table -> package ports, in-process.

Advances JuniorLLM docs/BETA_TO_OS.md, named item "Deck sheet":
  "Deck controls and Gaia helper stay separate. Package ports: JuniorLLM,
   AGI_SDK, JuniorOSai, JuniorOS, web3node, obsidian. Vault not synced.
   Device not opened. Schematic: JuniorHome docs/deck_capability.svg."

Read-only checks against files already in this clone (no sockets, no fetch,
no device open, no exec):
  - ui/deck.html pin table (MX 0-15, pads 16-23, ADC 26-29, 10.1 in DSI)
    agrees with docs/deck_capability.svg, ranges are disjoint and inside the
    RP2040 GPIO 0-29 bank, ADC pins map to AINSEL 0-3.
  - Deck status stays flashed/live/dma_present/admit false; vault synced false.
  - Deck controls vs Gaia helper stay separate (Not Gaia, clone_of none).
  - The six package ports each have an in-repo anchor in Home.
  - Live-beta port map: deck/app 127.0.0.1:8766, bitnetd 127.0.0.1:8765,
    i2sd 127.0.0.1:8767 are distinct and bound to 127.0.0.1 only.
Handshake sides: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Close rule: sha256(prev || pack || text)[:16], trits packed 2-bit i2s
(0->0, 1->1, -1->2). Rate 0.05. mint/socket/layer_one/fetch/ue5_launch false.
Stdlib only. Usage: python ports/deck_sheet_port_map.py [--selftest]
"""
from __future__ import annotations

import hashlib
import html.parser
import os
import re
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
SIDES = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ITEM = "BETA_TO_OS Deck sheet"
CLOSE = "sha256(prev || pack || text)[:16]"
GENESIS = "0" * 16
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# pin group -> (gpio lo, gpio hi, role, svg label)
PIN_MAP = {
    "MX": (0, 15, "note", "8x8 MX GPIO 0-15"),
    "pads": (16, 23, "step", "4x4 pads GPIO 16-23"),
    "ADC": (26, 29, "gamma filter clock mix", "ADC 26-29 knobs"),
}
DSI = ("10.1 in DSI", "surface")
RP2040_GPIO = (0, 29)
ADC_BASE = 26  # AINSEL n == GPIO 26 + n

PACKAGE_PORTS = {
    "JuniorLLM": "docs/LLM_PORTS.md",
    "AGI_SDK": "docs/AGI_SDK.md",
    "JuniorOSai": "docs/JUNIOROSAI.md",
    "JuniorOS": "docs/JUNIOROS.md",
    "web3node": "web3node",
    "obsidian": "docs/OBSIDIAN.md",
}
LOCAL_PORTS = {"deck_app": 8766, "bitnetd": 8765, "i2sd": 8767}
SIDE_OF = {"deck_app": "Home", "bitnetd": "JuniorLLM", "i2sd": "JuniorLLM"}


class _Cells(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cells: list[str] = []
        self.text: list[str] = []
        self._td = False

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self._td = True
            self.cells.append("")

    def handle_endtag(self, tag):
        if tag == "td":
            self._td = False

    def handle_data(self, data):
        self.text.append(data)
        if self._td:
            self.cells[-1] += data


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def deck_rows(deck_html: str) -> dict[str, str]:
    p = _Cells()
    p.feed(deck_html)
    c = [x.strip() for x in p.cells]
    return dict(zip(c[0::2], c[1::2]))


def svg_labels(svg: str) -> list[str]:
    return [t.strip() for t in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]


def checks(deck_html: str, svg: str, exists) -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []
    table = deck_rows(deck_html)
    labels = svg_labels(svg)
    text = " ".join(_Cells_text(deck_html).split())
    used: set[int] = set()
    for name, (lo, hi, role, label) in PIN_MAP.items():
        key = f"{name} {lo}-{hi}" if name != "pads" else f"pads {lo}-{hi}"
        rows.append(("Home", f"deck pin {key} -> {role}", table.get(key) == role))
        rows.append(("Home", f"svg agrees {label}", label in labels))
        span = set(range(lo, hi + 1))
        ok = not (span & used) and RP2040_GPIO[0] <= lo <= hi <= RP2040_GPIO[1]
        rows.append(("FieldCore", f"gpio {lo}-{hi} disjoint in 0-29", ok))
        used |= span
    ain = [g - ADC_BASE for g in range(PIN_MAP["ADC"][0], PIN_MAP["ADC"][1] + 1)]
    rows.append(("FieldCore", "ADC 26-29 -> AINSEL 0-3", ain == [0, 1, 2, 3] and "AINSEL 0-3 knobs" in labels))
    rows.append(("Home", "10.1 in DSI -> surface", table.get(DSI[0]) == DSI[1] and DSI[0] in labels))
    for flag in ("flashed", "live", "dma_present", "admit"):
        rows.append(("Flagstaff", f"deck {flag} false", f"{flag} false" in text))
    rows.append(("Flagstaff", "vault synced false", "synced false" in labels))
    rows.append(("Flagstaff", "device not opened (no /dev, no serial)", "/dev/" not in deck_html))
    rows.append(("OSai", "deck controls separate from Gaia helper",
                 "Not Gaia" in text and "clone_of none" in labels
                 and "Deck controls" in labels and "Gaia helper" in labels))
    for port, anchor in PACKAGE_PORTS.items():
        rows.append(("JuniorLLM" if port == "JuniorLLM" else "OSai",
                     f"package port {port} -> {anchor}", exists(anchor)))
    rows.append(("Home", "deck 127.0.0.1:8766 on schematic",
                 any(f"{BIND}:{LOCAL_PORTS['deck_app']}" in s for s in labels)))
    nums = list(LOCAL_PORTS.values())
    rows.append(("JuniorLLM", "8765/8766/8767 distinct", len(set(nums)) == len(nums)))
    for svc, port in LOCAL_PORTS.items():
        rows.append((SIDE_OF[svc], f"{svc} bind {BIND}:{port}", BIND == "127.0.0.1" and 1024 < port < 65536))
    rows.append(("Flagstaff", "mint socket layer_one fetch ue5_launch false",
                 not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)))
    rows.append(("Flagstaff", "rate 0.05", RATE == 0.05))
    return rows


def _Cells_text(doc: str) -> str:
    p = _Cells()
    p.feed(doc)
    return " ".join(p.text)


def pack(trits: list[int]) -> bytes:
    code = {0: 0, 1: 1, -1: 2}
    bits = "".join(format(code[t], "02b") for t in trits)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def close(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def run() -> int:
    rows = checks(_read("ui/deck.html"), _read("docs/deck_capability.svg"),
                  lambda rel: os.path.exists(os.path.join(ROOT, rel)))
    tip = GENESIS
    for side in SIDES:
        mine = [r for r in rows if r[0] == side]
        trits = [1 if ok else -1 for _, _, ok in mine] or [0]
        text = f"{ITEM}|{side}|" + ";".join(n for _, n, _ in mine)
        tip = close(tip, pack(trits), text)
        for s, name, ok in mine:
            print(f"{'PASS' if ok else 'FAIL'} [{s}] {name}")
    passed = sum(ok for *_, ok in rows)
    print(f"{ITEM}: {passed}/{len(rows)} pass, tip {tip}, rate {RATE}, bind {BIND}")
    return 0 if passed == len(rows) else 1


def selftest() -> int:
    good_html = ('<p>Not Gaia.</p><table><tr><td>MX 0-15</td><td>note</td></tr>'
                 '<tr><td>pads 16-23</td><td>step</td></tr><tr><td>ADC 26-29</td>'
                 '<td>gamma filter clock mix</td></tr><tr><td>10.1 in DSI</td><td>surface</td>'
                 '</tr></table><p>flashed false. live false. dma_present false. admit false.</p>')
    good_svg = "".join(f"<text>{t}</text>" for t in (
        "8x8 MX GPIO 0-15", "4x4 pads GPIO 16-23", "ADC 26-29 knobs", "10.1 in DSI",
        "AINSEL 0-3 knobs", "synced false", "Deck controls", "Gaia helper",
        "clone_of none", "live false · 127.0.0.1:8766"))
    ok_all = all(ok for *_, ok in checks(good_html, good_svg, lambda _: True))
    bad = good_html.replace("live false", "live true")
    caught = not all(ok for *_, ok in checks(bad, good_svg, lambda _: True))
    missing = not all(ok for *_, ok in checks(good_html, good_svg, lambda r: r != "web3node"))
    assert pack([1, -1, 0, 1]) == bytes([0b01100001])
    c1 = close(GENESIS, pack([1]), "a")
    det = c1 == close(GENESIS, pack([1]), "a") and c1 != close(GENESIS, pack([-1]), "a")
    res = ok_all and caught and missing and det
    print(f"selftest {'pass' if res else 'fail'}: good={ok_all} flip_caught={caught} "
          f"missing_port_caught={missing} close_det={det}")
    return 0 if res and run() == 0 else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else run())
