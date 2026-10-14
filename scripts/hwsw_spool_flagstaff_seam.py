#!/usr/bin/env python3
"""Hardware-to-software seam emulator: lab spool + device catalog -> Flagstaff FieldCore row.

What it emulates
  A job file dropped on the lab file-spool (ports/lab_spool.py) is checked
  against the OSS device catalog (ports/lab_catalog.py) and turned into one
  FieldCore intent/access label row of the kind Flagstaff keeps
  (docs/FIELDCORE_FLAGSTAFF.md). Rows are chained with the shared close rule
  sha256(prev || pack || text)[:16] and carry the shared rate 0.05.

What it does not touch
  No sockets, no real devices, no printers, lasers, GPIO or ROS. Nothing under
  ~/.juniorhome: the spool and the JSONL row live in a temp dir that is removed
  after the run. No network, no remote code, no model or GGUF. Standard library
  only. Flags: mint, socket, layer_one, fetch, ue5_launch all False; bind
  127.0.0.1 is reported, never opened.

How to run
  python3 scripts/hwsw_spool_flagstaff_seam.py          # self-test, prints JSON
  python3 scripts/hwsw_spool_flagstaff_seam.py demo     # one emulated job
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

RATE = 0.05
BIND = "127.0.0.1"
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
GENESIS = "0" * 16

# Mirror of ports/lab_catalog.py; the live catalog is used when importable.
_FALLBACK_CATALOG = [
    {"id": "klipper-moonraker", "kind": "fdm", "port": 7125, "files": [".gcode"]},
    {"id": "octoprint", "kind": "fdm", "port": 5000, "files": [".gcode"]},
    {"id": "grbl-laser", "kind": "laser", "port": None, "files": [".nc", ".gcode"]},
    {"id": "lightburn-export", "kind": "laser", "port": None, "files": [".lbrn2"]},
    {"id": "cups-ipp", "kind": "paper", "port": 631, "files": [".pdf", ".ps"]},
    {"id": "pi-gpio", "kind": "sbc", "port": None, "files": []},
    {"id": "ros2-loopback", "kind": "robot", "port": None, "files": []},
    {"id": "shop-screen", "kind": "calc", "port": None, "files": [".json"]},
]


def catalog() -> list:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from ports.lab_catalog import CATALOG  # local file read only
        return list(CATALOG)
    except Exception:
        return list(_FALLBACK_CATALOG)


def close(prev: str, pack: str, text: str) -> str:
    return hashlib.sha256((prev + pack + text).encode("utf-8")).hexdigest()[:16]


# Device kind -> FieldCore intent label (labels only, never actions).
INTENT = {"fdm": "make", "laser": "make", "paper": "print", "sbc": "sense",
          "robot": "move", "calc": "show"}


def spool(root: Path, device: str, name: str, body: str) -> dict:
    ids = {c["id"]: c for c in catalog()}
    if device not in ids:
        return {"ok": False, "device": device, "reason": "not in catalog"}
    exts = ids[device]["files"]
    if exts and Path(name).suffix not in exts:
        return {"ok": False, "device": device, "reason": f"suffix not in {exts}"}
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{device}-{name}"
    dest.write_text(body, encoding="utf-8")
    return {"ok": True, "path": str(dest), "fire": False, "bind": BIND}


def to_fieldcore(root: Path, lake: Path, prev: str = GENESIS) -> list:
    ids = {c["id"]: c for c in catalog()}
    rows = []
    for p in sorted(x for x in root.iterdir() if x.is_file()):
        dev = next((d for d in ids if p.name.startswith(d + "-")), None)
        if dev is None:
            continue
        kind = ids[dev]["kind"]
        text = p.read_text(encoding="utf-8")
        pack = json.dumps({"device": dev, "kind": kind, "file": p.name}, sort_keys=True)
        sha = close(prev, pack, text)
        rows.append({"node": "flagstaff", "device": dev, "kind": kind,
                     "intent": INTENT.get(kind, "unknown"), "bytes": len(text.encode()),
                     "rate": RATE, "prev": prev, "close": sha, "fire": False, **FLAGS})
        prev = sha
    with lake.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return rows


def self_test() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root, lake = Path(td) / "spool", Path(td) / "fieldcore_flagstaff.jsonl"
        a = spool(root, "klipper-moonraker", "cube.gcode", "G28\nG1 X10\n")
        b = spool(root, "cups-ipp", "bom.pdf", "%PDF-emu\n")
        bad_dev = spool(root, "unknown-cnc", "x.nc", "")
        bad_ext = spool(root, "grbl-laser", "x.pdf", "")
        rows = to_fieldcore(root, lake)
        assert a["ok"] and b["ok"] and not bad_dev["ok"] and not bad_ext["ok"]
        assert len(rows) == 2 and rows[0]["prev"] == GENESIS
        assert rows[1]["prev"] == rows[0]["close"]
        for r in rows:
            pack = json.dumps({"device": r["device"], "kind": r["kind"],
                               "file": f"{r['device']}-" + ("bom.pdf" if r["kind"] == "paper" else "cube.gcode")},
                              sort_keys=True)
            body = (root / json.loads(pack)["file"]).read_text(encoding="utf-8")
            assert r["close"] == close(r["prev"], pack, body)
            assert r["rate"] == RATE and r["fire"] is False
            assert not any(r[k] for k in FLAGS)
        assert len(lake.read_text(encoding="utf-8").splitlines()) == 2
        assert {r["intent"] for r in rows} == {"make", "print"}
    return {"ok": True, "rows": 2, "rate": RATE, "bind": BIND, **FLAGS, "fire": False}


def demo() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "spool"
        spool(root, "shop-screen", "board.json", '{"panel": "dash"}\n')
        return {"rows": to_fieldcore(root, Path(td) / "row.jsonl")}


if __name__ == "__main__":
    out = demo() if (len(sys.argv) > 1 and sys.argv[1] == "demo") else self_test()
    print(json.dumps(out, indent=2))
