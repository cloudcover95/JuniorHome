"""JuniorDeck class map. Does not boot a vendor OS."""
from __future__ import annotations

import json
from pathlib import Path

TOML = Path(__file__).resolve().parents[1] / "config" / "deck_board.toml"
OUT = Path.home() / ".juniorhome" / "os" / "deck_board.json"


def load() -> dict:
    text = TOML.read_text(encoding="utf-8")
    body = {
        "protocol": "goldend-osai-omega/1",
        "name": "JuniorDeck",
        "kept": ["audio_drop", "trit_pack", "geode_grid", "gamma"],
        "added": ["pads", "pad_banks", "keys", "wheels", "knobs", "pedals", "line_io", "cv_gate", "midi_din", "usb_audio_class"],
        "excluded": ["vendor-os", "q-link", "live-control", "project-import", "quadrant-pad"],
        "live": False,
        "license": "MIT",
        "patent_note": "quadrant pressure pad is pending elsewhere; 4x4 velocity pads and USB audio class are standards",
        "bind": "127.0.0.1",
        "toml": text.count("usb_audio_class") > 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return body
