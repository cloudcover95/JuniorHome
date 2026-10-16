#!/usr/bin/env python3
"""Hardware-to-software seam emulator: SBC (arm64, ≤12 W) -> JuniorOS T4 / OSai overlay row.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md — JuniorOS overlay (Debian/Alpine + os-release.junior
  + hardened bitnetd.service + juniorctl on PATH). Hardware class from
  JuniorHome docs/HW_CLASS.md: SBC → JuniorOS T4. Not ROADMAP T13 (overnight).

What it emulates
  An in-memory SBC class node (arm64, watts_claim ≤12) is gated into a JuniorOS
  T4_ship envelope and one OSai note-pack row. os-release.junior identity fields
  and a bitnetd bind intent of 127.0.0.1:8765 are reported only; juniorctl_on_path
  is a boolean gate, not a real PATH probe of another process. Trit pack is a
  short byte string derived from arch/watt markers (codes 0/1/2 packed three per
  byte). Rows use close rule sha256(prev || pack || text)[:16] with pack as real
  bytes, and carry rate 0.05. overlay_ready may become true; live stays false.

What it does not touch
  No real GPIO, /sys, /proc, SSH, or flash of any board. No sockets opened to
  bitnetd, i2sd, or devices (bind string only). No Asahi probe, no SFF/T0 path,
  no Deck USB classes, no spool. Nothing under ~/.juniorhome: inventory and rows
  live in a tempfile removed after the run. Standard library only. Flags: mint,
  socket, layer_one, fetch, ue5_launch all False.

How to run
  python3 scripts/hwsw_sbc_junioros_t4_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_sbc_junioros_t4_seam.py demo     # one emulated T4/OSai row
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

RATE = 0.05
BIND = "127.0.0.1"
BITNETD_PORT = 8765
WATT_CAP = 12.0
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16

# Overlay gate labels (BETA_TO_OS JuniorOS overlay; labels only).
OVERLAY6 = (
    "arch_arm64",
    "watts_le_12",
    "os_release_junior",
    "bitnetd_bind_local",
    "juniorctl_on_path",
    "osai_note_ready",
)

# Emulated SBC marker bytes (stdlib only; not real board sensors).
_EMU_MARKERS = bytes([8, 12, 4, 11, 9, 7, 10, 6, 5, 12, 3, 8])


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def trit_pack(samples: bytes) -> bytes:
    """Map sample values into trit codes 0/1/2, pack three trits per byte."""
    codes = []
    for s in samples:
        if s < 5:
            codes.append(0)
        elif s < 10:
            codes.append(1)
        else:
            codes.append(2)
    out = bytearray()
    for i in range(0, len(codes), 3):
        chunk = codes[i : i + 3]
        while len(chunk) < 3:
            chunk.append(0)
        out.append(chunk[0] + 3 * chunk[1] + 9 * chunk[2])
    return bytes(out)


def inventory(nodes: list[dict] | None = None) -> list[dict]:
    """In-memory emulated SBC class inventory (docs/HW_CLASS.md)."""
    if nodes is not None:
        return list(nodes)
    return [
        {
            "id": "sbc-limb-01",
            "class": "SBC",
            "arch": "arm64",
            "watts_claim": 8.0,
            "owner": "JuniorOS T4",
            "present": True,
        }
    ]


def sbc_class_ok(node: dict) -> bool:
    return (
        node.get("present") is True
        and node.get("class") == "SBC"
        and node.get("arch") == "arm64"
        and float(node.get("watts_claim", 99.0)) <= WATT_CAP
    )


def overlay6_and(gates: dict[str, bool]) -> bool:
    return all(gates.get(name, False) for name in OVERLAY6)


def os_release_junior(node: dict) -> dict:
    """Emulated os-release.junior identity (in-memory only)."""
    return {
        "NAME": "JuniorOS",
        "ID": "junioros",
        "VARIANT": "overlay",
        "VARIANT_ID": "t4_ship",
        "ARCH": node.get("arch", "unknown"),
        "JUNIOR_OWNER": "JuniorOS T4",
        "JUNIOR_ENVELOPE": "t4",
    }


def bitnetd_bind_intent() -> dict:
    """Report bind intent only; never open a socket."""
    return {"host": BIND, "port": BITNETD_PORT, "opened": False}


def overlay_ready(node: dict, gates: dict[str, bool]) -> bool:
    return sbc_class_ok(node) and overlay6_and(gates)


def default_gates(ok: bool = True) -> dict[str, bool]:
    return {name: ok for name in OVERLAY6}


def build_row(
    prev: str,
    pack: bytes,
    text: str,
    ready: bool,
    lake: Path | None = None,
) -> dict:
    sha = close(prev, pack, text)
    row = {
        "node": "junioros_t4",
        "kind": "osai_note",
        "envelope": "t4",
        "class": "SBC",
        "rate": RATE,
        "prev": prev,
        "close": sha,
        "fire": False,
        "bind": BIND,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "overlay_ready": ready,
        "live": False,
        **FLAGS,
    }
    if lake is not None:
        lake.parent.mkdir(parents=True, exist_ok=True)
        with lake.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def self_test() -> dict:
    with tempfile.TemporaryDirectory() as td:
        lake = Path(td) / "osai_inbox" / "t4_notes.jsonl"
        pack = trit_pack(_EMU_MARKERS)
        assert isinstance(pack, bytes) and len(pack) > 0

        nodes = inventory()
        node = nodes[0]
        assert sbc_class_ok(node) is True
        gates = default_gates(True)
        ready = overlay_ready(node, gates)
        assert ready is True

        release = os_release_junior(node)
        assert release["VARIANT_ID"] == "t4_ship"
        bind = bitnetd_bind_intent()
        assert bind["host"] == BIND and bind["opened"] is False

        text = json.dumps(
            {
                "sbc": {"id": node["id"], "arch": node["arch"], "watts": node["watts_claim"]},
                "os_release": release,
                "bitnetd": bind,
                "pack": pack.hex(),
            },
            sort_keys=True,
        )
        row = build_row(GENESIS, pack, text, ready, lake)
        assert row["overlay_ready"] is True and row["live"] is False
        assert row["rate"] == RATE and row["fire"] is False
        assert row["prev"] == GENESIS
        assert row["close"] == close(GENESIS, pack, text)
        assert not any(row[k] for k in FLAGS)
        assert row["bind"] == BIND
        assert row["envelope"] == "t4"

        # Over-budget watts -> class fail.
        fat = dict(node, watts_claim=15.0)
        assert sbc_class_ok(fat) is False
        assert overlay_ready(fat, gates) is False

        # Wrong arch -> class fail.
        x86 = dict(node, arch="x86_64")
        assert sbc_class_ok(x86) is False

        # Broken overlay gate -> not ready.
        broken = default_gates(True)
        broken["juniorctl_on_path"] = False
        assert overlay_ready(node, broken) is False

        # Absent node -> not ready.
        missing = [dict(node, present=False)]
        assert overlay_ready(missing[0], gates) is False

        # close() verifies with trit bytes pack (not a JSON label string).
        assert close(GENESIS, pack, text) == row["close"]
        wrong = close(GENESIS, b"not-trit", text)
        assert wrong != row["close"]

        assert lake.is_file()
        lines = lake.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    return {
        "ok": True,
        "overlay_ready": True,
        "live": False,
        "rate": RATE,
        "bind": BIND,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "envelope": "t4",
        "pack_len": len(pack),
        **FLAGS,
        "fire": False,
    }


def demo() -> dict:
    with tempfile.TemporaryDirectory() as td:
        lake = Path(td) / "osai_inbox" / "t4_notes.jsonl"
        pack = trit_pack(_EMU_MARKERS)
        nodes = inventory()
        node = nodes[0]
        gates = default_gates(True)
        ready = overlay_ready(node, gates)
        release = os_release_junior(node)
        bind = bitnetd_bind_intent()
        text = json.dumps(
            {
                "sbc": {
                    "id": node["id"],
                    "arch": node["arch"],
                    "watts": node["watts_claim"],
                    "owner": node["owner"],
                },
                "os_release": release,
                "bitnetd": bind,
                "pack": pack.hex(),
            },
            sort_keys=True,
        )
        row = build_row(GENESIS, pack, text, ready, lake)
        return {"row": row, "pack_hex": pack.hex(), "lake": str(lake), "os_release": release}


if __name__ == "__main__":
    out = demo() if (len(sys.argv) > 1 and sys.argv[1] == "demo") else self_test()
    print(json.dumps(out, indent=2))
