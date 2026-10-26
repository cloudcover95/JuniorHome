#!/usr/bin/env python3
"""Hardware-to-software seam emulator: USB Audio Class stream format (/proc/asound/cardN/stream0) -> JuniorDeck PCM16 inbox admit.

Roadmap item
  JuniorLLM docs/BETA_TO_OS.md "Deck sheet" (Deck sheet package ports; USB
  sound goes to JuniorDeck). JuniorHome docs/DECK_AUDIO.md says the Deck reads
  WAV with stdlib `wave` as PCM16 and caps a pass at 3600 s. This file covers
  the seam that the earlier presence check (USB class 01 seen or not) leaves
  open: which capture format/rate/channel layout a USB sound card actually
  offers, and whether that lands as PCM16 the Deck inbox can read.

What it emulates
  Text in the shape of /proc/asound/cards and /proc/asound/cardN/stream0 as
  the snd-usb-audio driver prints it (Playback/Capture sections, Interface /
  Altset / Format / Channels / Rates lines, discrete or "a - b" continuous
  rates). The parser picks a capture altset and returns one decision:
    admit         S16_LE capture, 1-2 ch (or >2 ch with first-two mapping
                  flagged), rate 48000 preferred, else 44100
    hold_convert  capture exists but only S24_3LE / S24_LE / S32_LE / FLOAT
                  (Deck wave reader is PCM16; no converter is assumed local)
    hold_rate     S16_LE present but neither 48000 nor 44100 offered
    hold_playback_only  device has no Capture section
    reject_not_usb      card driver is not USB-Audio (onboard codecs stay out)
    hold_unparsed       stream text could not be parsed
  For admitted rows it reports bytes per 3600 s pass and flags stride when
  the pass would exceed 512 MiB. Rows chain with close rule
  sha256(prev || pack || text)[:16], pack = trit codes of the decision
  (0 hold, 1 admit, 2 reject) packed 4 per byte, rate 0.05.

What it does not touch
  No /proc/asound, /dev/snd, /dev/bus/usb or ALSA library is read or opened;
  every stream text is an inline fixture. No sockets: the Deck bind
  127.0.0.1:8766 is reported as a string only. No ffmpeg, no model, no GGUF,
  no network, nothing under ~/.juniorhome. Standard library only.
  Flags: mint, socket, layer_one, fetch, ue5_launch all False.

How to run
  python3 scripts/hwsw_alsa_stream_format_deck_pcm16_seam.py         # self-test, prints JSON
  python3 scripts/hwsw_alsa_stream_format_deck_pcm16_seam.py demo    # one emulated row
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

RATE = 0.05
DECK_BIND = "127.0.0.1:8766"
PASS_CAP_S = 3600
STRIDE_BYTES = 512 * 1024 * 1024
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
TRIT = {"admit": 1, "reject_not_usb": 2}
PCM16 = "S16_LE"
WIDE = {"S24_3LE": 3, "S24_LE": 4, "S32_LE": 4, "FLOAT_LE": 4}


def close(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode("utf-8") + pack + text.encode("utf-8")).hexdigest()[:16]


def trit_pack(codes: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(codes), 4):
        b = 0
        for j, c in enumerate(codes[i:i + 4]):
            b |= (c & 3) << (2 * j)
        out.append(b)
    return bytes(out)


def card_driver(cards_text: str, card: int) -> str | None:
    m = re.search(r"^\s*%d\s+\[[^\]]*\]:\s*(\S+)" % card, cards_text, re.M)
    return m.group(1) if m else None


def parse_rates(s: str) -> tuple[list[int], tuple[int, int] | None]:
    s = s.strip()
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)(?:\s*\(.*\))?", s)
    if m:
        return [], (int(m.group(1)), int(m.group(2)))
    vals = [int(x) for x in re.findall(r"\d+", s)]
    return vals, None


def parse_stream(text: str) -> dict:
    """Return {'Playback': [altsets], 'Capture': [altsets]}; altset = dict(format, channels, rates, range)."""
    out: dict[str, list[dict]] = {"Playback": [], "Capture": []}
    section = None
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line in ("Playback:", "Capture:"):
            section = line[:-1]
            cur = None
            continue
        if section is None:
            continue
        if line.startswith("Altset"):
            cur = {"format": None, "channels": None, "rates": [], "range": None}
            out[section].append(cur)
        elif cur is not None and line.startswith("Format:"):
            cur["format"] = line.split(":", 1)[1].strip().split()[0]
        elif cur is not None and line.startswith("Channels:"):
            cur["channels"] = int(line.split(":", 1)[1].strip())
        elif cur is not None and line.startswith("Rates:"):
            cur["rates"], cur["range"] = parse_rates(line.split(":", 1)[1])
    for alts in out.values():
        for a in alts:
            if a["format"] is None or a["channels"] is None or not (a["rates"] or a["range"]):
                raise ValueError("incomplete altset")
    if not out["Playback"] and not out["Capture"]:
        raise ValueError("no sections")
    return out


def offers(alt: dict, rate: int) -> bool:
    if rate in alt["rates"]:
        return True
    r = alt["range"]
    return bool(r and r[0] <= rate <= r[1])


def decide(cards_text: str, card: int, stream_text: str) -> dict:
    drv = card_driver(cards_text, card)
    if drv != "USB-Audio":
        return {"decision": "reject_not_usb", "driver": drv}
    try:
        st = parse_stream(stream_text)
    except (ValueError, TypeError):
        return {"decision": "hold_unparsed"}
    cap = st["Capture"]
    if not cap:
        return {"decision": "hold_playback_only"}
    s16 = [a for a in cap if a["format"] == PCM16]
    if not s16:
        return {"decision": "hold_convert", "formats": sorted({a["format"] for a in cap})}
    for rate in (48000, 44100):
        cands = [a for a in s16 if offers(a, rate)]
        if cands:
            alt = min(cands, key=lambda a: (a["channels"] > 2, abs(a["channels"] - 2)))
            ch = alt["channels"]
            used = min(ch, 2)
            nbytes = rate * 2 * used * PASS_CAP_S
            return {
                "decision": "admit",
                "format": PCM16,
                "rate": rate,
                "channels": ch,
                "mapped_channels": used,
                "first_two_mapped": ch > 2,
                "pass_bytes": nbytes,
                "stride": nbytes > STRIDE_BYTES,
            }
    return {"decision": "hold_rate"}


def row(prev: str, name: str, d: dict) -> dict:
    code = TRIT.get(d["decision"], 0)
    pack = trit_pack([code, 1 if d.get("stride") else 0, 1 if d.get("first_two_mapped") else 0])
    text = json.dumps({"fixture": name, **d, "deck_bind": DECK_BIND, "rate": RATE}, sort_keys=True)
    return {"fixture": name, **d, "close": close(prev, pack, text)}


CARDS = """ 0 [PCH            ]: HDA-Intel - HDA Intel PCH
                      HDA Intel PCH at 0xf7f10000 irq 31
 1 [Device         ]: USB-Audio - USB Audio Device
                      Generic USB Audio at usb-0000:00:14.0-2, full speed
"""

STEREO_48K = """Generic USB Audio at usb-0000:00:14.0-2, full speed : USB Audio

Playback:
  Status: Stop
  Interface 1
    Altset 1
    Format: S16_LE
    Channels: 2
    Endpoint: 0x01 (1 OUT) (ADAPTIVE)
    Rates: 44100, 48000

Capture:
  Status: Stop
  Interface 2
    Altset 1
    Format: S16_LE
    Channels: 1
    Endpoint: 0x82 (2 IN) (ASYNC)
    Rates: 44100, 48000
"""

WIDE_ONLY = """X : USB Audio

Capture:
  Interface 2
    Altset 1
    Format: S24_3LE
    Channels: 2
    Rates: 44100, 48000, 96000
    Altset 2
    Format: S32_LE
    Channels: 2
    Rates: 8000 - 192000 (continuous)
"""

RANGE_4CH = """X : USB Audio

Capture:
  Interface 3
    Altset 1
    Format: S16_LE
    Channels: 4
    Rates: 8000 - 48000 (continuous)
"""

ODD_RATE = """X : USB Audio

Capture:
  Interface 2
    Altset 1
    Format: S16_LE
    Channels: 2
    Rates: 16000, 32000
"""

PLAY_ONLY = """X : USB Audio

Playback:
  Interface 1
    Altset 1
    Format: S16_LE
    Channels: 2
    Rates: 48000
"""

BROKEN = """X : USB Audio

Capture:
  Interface 2
    Altset 1
    Channels: two
"""

FIXTURES = [
    ("usb_mono_48k", 1, STEREO_48K, {"decision": "admit", "rate": 48000, "mapped_channels": 1, "stride": False}),
    ("usb_wide_only", 1, WIDE_ONLY, {"decision": "hold_convert"}),
    ("usb_range_4ch", 1, RANGE_4CH, {"decision": "admit", "rate": 48000, "mapped_channels": 2, "first_two_mapped": True, "stride": True}),
    ("usb_odd_rate", 1, ODD_RATE, {"decision": "hold_rate"}),
    ("usb_playback_only", 1, PLAY_ONLY, {"decision": "hold_playback_only"}),
    ("usb_unparsed", 1, BROKEN, {"decision": "hold_unparsed"}),
    ("onboard_hda", 0, STEREO_48K, {"decision": "reject_not_usb"}),
]


def self_test() -> dict:
    prev = "0" * 16
    rows, ok = [], True
    for name, card, text, want in FIXTURES:
        d = decide(CARDS, card, text)
        r = row(prev, name, d)
        prev = r["close"]
        good = all(d.get(k) == v for k, v in want.items())
        ok &= good
        rows.append({**r, "pass": good})
    assert DECK_BIND.startswith("127.0.0.1:") and not any(FLAGS.values())
    assert close("a", b"\x01", "t") == hashlib.sha256(b"a\x01t").hexdigest()[:16]
    return {"ok": ok, "rows": rows, "tip_close": prev, "flags": FLAGS, "rate": RATE, "deck_bind": DECK_BIND}


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "demo":
        print(json.dumps(row("0" * 16, "demo", decide(CARDS, 1, STEREO_48K)), indent=2))
        return 0
    res = self_test()
    print(json.dumps(res, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
