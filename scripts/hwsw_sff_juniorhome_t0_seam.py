#!/usr/bin/env python3
"""Hardware-to-software seam emulator: SFF host (x86_64 or arm64) -> JuniorHome T0 home clock.

Roadmap item
  JuniorHome docs/ROADMAP.md #4 "Asahi MLX — probe only from automations",
  on the hardware class from docs/HW_CLASS.md: SFF host -> JuniorHome T0.
  The seam decides which compute clock an emulated T0 host may run. An arm64
  Asahi host can report MLX as seen, but the admit stays False (probe only);
  the T0 Linux clock stays numpy (docs/NUMPY_CLOCK.md), and there is no UE boot
  on T0 (docs/HOME_KERNEL.md). Not ROADMAP T13 (overnight owns it).

What it emulates
  In-memory SFF host fixtures (arch, OS id, Asahi device-tree marker string,
  local disk flag, an "mlx importable" boolean, watts claim). Nothing is
  imported or probed: mlx_seen is the fixture value, never `import mlx`.
  The host is class-gated (SFF host, x86_64 or arm64, local disk, present),
  then mapped to a T0 clock decision: backend "numpy", mlx_admit False,
  ue_boot False, bitnetd 127.0.0.1:8765 and i2sd 127.0.0.1:8767 as bind
  strings only. One T0 row is written with close rule
  sha256(prev || pack || text)[:16], where pack is real trit bytes from
  emulated per-core load markers (codes 0/1/2, three per byte), at rate 0.05.
  t0_ready may become true; live stays false.

What it does not touch
  No /proc, /sys, device tree, SSH, firmware, or Asahi installer. No socket to
  bitnetd, i2sd, or any device (bind strings only). No MLX, numpy, or CUDA
  import. No SBC/T4 path, Deck USB classes, or spool (earlier seams), and it
  does not wrap or re-run any ports/ handshake or other team file. Nothing under
  ~/.juniorhome: rows go to a tempfile removed after the run. Standard library
  only. Flags: mint, socket, layer_one, fetch, ue5_launch all False.

How to run
  python3 scripts/hwsw_sff_juniorhome_t0_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_sff_juniorhome_t0_seam.py demo     # emulated T0 rows (Asahi + x86_64)
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
I2SD_PORT = 8767
SFF_ARCHES = ("x86_64", "arm64")
FLAGS = {
    "mint": False,
    "socket": False,
    "layer_one": False,
    "fetch": False,
    "ue5_launch": False,
}
GENESIS = "0" * 16

# T0 gate labels (labels only; every one must hold for t0_ready).
T0_GATES = (
    "sff_class",
    "local_disk",
    "numpy_clock",
    "mlx_probe_only",
    "no_ue_boot",
    "bind_local",
)

# Emulated per-core load markers, 0..15 (not real sensors).
_EMU_LOAD = bytes([3, 9, 14, 6, 1, 11, 7, 4, 13, 8, 2, 10, 5, 12])


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def trit_pack(samples: bytes) -> bytes:
    """Map load markers into trit codes 0/1/2, pack three trits per byte."""
    codes = [0 if s < 5 else (1 if s < 10 else 2) for s in samples]
    out = bytearray()
    for i in range(0, len(codes), 3):
        chunk = codes[i : i + 3] + [0] * (3 - len(codes[i : i + 3]))
        out.append(chunk[0] + 3 * chunk[1] + 9 * chunk[2])
    return bytes(out)


def trit_unpack(pack: bytes, n: int) -> list[int]:
    codes: list[int] = []
    for b in pack:
        codes.extend((b % 3, (b // 3) % 3, (b // 9) % 3))
    return codes[:n]


def fixtures() -> list[dict]:
    """In-memory emulated SFF host class rows (docs/HW_CLASS.md, docs/HOMELAB.md)."""
    return [
        {
            "id": "sff-home-asahi",
            "class": "SFF host",
            "arch": "arm64",
            "os_id": "fedora-asahi-remix",
            "dt_compatible": "apple,j473",  # fixture string, never read from /proc
            "local_disk": True,
            "mlx_importable": True,  # fixture value, never imported
            "watts_claim": 45.0,
            "present": True,
        },
        {
            "id": "sff-home-x86",
            "class": "SFF host",
            "arch": "x86_64",
            "os_id": "debian",
            "dt_compatible": "",
            "local_disk": True,
            "mlx_importable": False,
            "watts_claim": 35.0,
            "present": True,
        },
    ]


def sff_class_ok(host: dict) -> bool:
    return (
        host.get("present") is True
        and host.get("class") == "SFF host"
        and host.get("arch") in SFF_ARCHES
    )


def asahi_probe(host: dict) -> dict:
    """Probe-only report (ROADMAP #4). Reads the fixture dict, nothing else."""
    asahi = host.get("arch") == "arm64" and str(host.get("dt_compatible", "")).startswith("apple,")
    return {
        "asahi": asahi,
        "mlx_seen": bool(asahi and host.get("mlx_importable") is True),
        "mlx_admit": False,
        "source": "fixture",
    }


def t0_clock(host: dict, probe: dict, bind: str = BIND, want_ue: bool = False,
             want_mlx: bool = False) -> dict:
    """T0 home-clock decision. MLX and UE requests are refused, not honored."""
    return {
        "backend": "numpy",
        "mlx_requested": want_mlx,
        "mlx_admit": False,
        "mlx_seen": probe["mlx_seen"],
        "ue_requested": want_ue,
        "ue_boot": False,
        "bind": bind,
        "bitnetd": {"host": bind, "port": BITNETD_PORT, "opened": False},
        "i2sd": {"host": bind, "port": I2SD_PORT, "opened": False},
    }


def gates(host: dict, probe: dict, clock: dict) -> dict[str, bool]:
    return {
        "sff_class": sff_class_ok(host),
        "local_disk": host.get("local_disk") is True,
        "numpy_clock": clock["backend"] == "numpy",
        "mlx_probe_only": probe["mlx_admit"] is False and clock["mlx_admit"] is False
        and not clock["mlx_requested"],
        "no_ue_boot": clock["ue_boot"] is False and not clock["ue_requested"],
        "bind_local": clock["bind"] == BIND,
    }


def t0_ready(g: dict[str, bool]) -> bool:
    return all(g.get(name, False) for name in T0_GATES)


def build_row(prev: str, pack: bytes, host: dict, probe: dict, clock: dict,
              lake: Path | None = None) -> dict:
    g = gates(host, probe, clock)
    text = json.dumps(
        {"host": host["id"], "arch": host.get("arch"), "probe": probe,
         "clock": clock, "gates": g, "pack": pack.hex()},
        sort_keys=True,
    )
    row = {
        "node": "juniorhome_t0",
        "kind": "t0_clock",
        "class": "SFF host",
        "host": host["id"],
        "rate": RATE,
        "prev": prev,
        "close": close(prev, pack, text),
        "text": text,
        "fire": False,
        "bind": BIND,
        "backend": clock["backend"],
        "asahi": probe["asahi"],
        "mlx_seen": probe["mlx_seen"],
        "mlx_admit": False,
        "t0_ready": t0_ready(g),
        "live": False,
        **FLAGS,
    }
    if lake is not None:
        lake.parent.mkdir(parents=True, exist_ok=True)
        with lake.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def run_chain(lake: Path | None) -> list[dict]:
    pack = trit_pack(_EMU_LOAD)
    rows, prev = [], GENESIS
    for host in fixtures():
        probe = asahi_probe(host)
        row = build_row(prev, pack, host, probe, t0_clock(host, probe), lake)
        rows.append(row)
        prev = row["close"]
    return rows


def self_test() -> dict:
    with tempfile.TemporaryDirectory() as td:
        lake = Path(td) / "t0_inbox" / "t0_clock.jsonl"
        pack = trit_pack(_EMU_LOAD)
        assert isinstance(pack, bytes) and len(pack) == 5
        assert trit_unpack(pack, len(_EMU_LOAD)) == [
            0 if s < 5 else (1 if s < 10 else 2) for s in _EMU_LOAD
        ]

        rows = run_chain(lake)
        asahi_row, x86_row = rows
        assert asahi_row["asahi"] is True and asahi_row["mlx_seen"] is True
        assert x86_row["asahi"] is False and x86_row["mlx_seen"] is False
        for row in rows:
            assert row["t0_ready"] is True and row["live"] is False
            assert row["backend"] == "numpy" and row["mlx_admit"] is False
            assert row["rate"] == RATE and row["fire"] is False
            assert row["bind"] == BIND
            assert not any(row[k] for k in FLAGS)
        # Chain: second row links to the first; close recomputes from trit bytes.
        assert asahi_row["prev"] == GENESIS and x86_row["prev"] == asahi_row["close"]
        for row in rows:
            assert row["close"] == close(row["prev"], pack, row["text"])
            assert row["close"] != close(row["prev"], b"not-trit", row["text"])

        host = fixtures()[0]
        probe = asahi_probe(host)

        # MLX request on T0 is refused: gate fails, backend stays numpy.
        c = t0_clock(host, probe, want_mlx=True)
        assert c["backend"] == "numpy" and c["mlx_admit"] is False
        assert t0_ready(gates(host, probe, c)) is False

        # UE boot request on T0 is refused.
        assert t0_ready(gates(host, probe, t0_clock(host, probe, want_ue=True))) is False

        # Any non-loopback bind fails the bind gate.
        assert t0_ready(gates(host, probe, t0_clock(host, probe, bind="192.168.1.10"))) is False

        # Wrong arch (32-bit SBC-style) and wrong class fail the class gate.
        for bad in (dict(host, arch="armv7l"), dict(host, **{"class": "SBC"}),
                    dict(host, present=False)):
            p = asahi_probe(bad)
            assert t0_ready(gates(bad, p, t0_clock(bad, p))) is False

        # No local disk -> not ready (docs/HOMELAB.md: T0 has local disk).
        nodisk = dict(host, local_disk=False)
        assert t0_ready(gates(nodisk, probe, t0_clock(nodisk, probe))) is False

        # x86_64 with an apple dt string is still not Asahi.
        odd = dict(fixtures()[1], dt_compatible="apple,j473", mlx_importable=True)
        assert asahi_probe(odd)["asahi"] is False

        lines = lake.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    return {
        "ok": True,
        "rows": 2,
        "t0_ready": True,
        "backend": "numpy",
        "mlx_admit": False,
        "live": False,
        "rate": RATE,
        "bind": BIND,
        "bitnetd": f"{BIND}:{BITNETD_PORT}",
        "i2sd": f"{BIND}:{I2SD_PORT}",
        "pack_len": len(pack),
        "tip": rows[-1]["close"],
        **FLAGS,
        "fire": False,
    }


def demo() -> dict:
    with tempfile.TemporaryDirectory() as td:
        lake = Path(td) / "t0_inbox" / "t0_clock.jsonl"
        rows = run_chain(lake)
        return {"rows": rows, "pack_hex": trit_pack(_EMU_LOAD).hex(), "lake": str(lake)}


if __name__ == "__main__":
    out = demo() if (len(sys.argv) > 1 and sys.argv[1] == "demo") else self_test()
    print(json.dumps(out, indent=2))
