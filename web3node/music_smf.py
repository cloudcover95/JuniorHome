from pathlib import Path
from music_pattern import HAT, KICK, SNAR
NOTES = {"kick": 36, "snare": 38, "hat": 42}
PPQ, STEP = 24, 6
def _vlq(n):
    n = max(0, n); out = [n & 0x7F]; n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F)); n >>= 7
    return bytes(reversed(out))
def _track(rows, notes=None):
    notes = notes or NOTES
    ev, last, ons = bytearray(), 0, []
    for i in range(16):
        t = i * STEP
        for name, row in rows.items():
            if row[i] == "1":
                ons.append((t, notes[name], 100))
                ons.append((t + STEP - 1, notes[name], 0))
    ons.sort()
    for t, note, vel in ons:
        ev += _vlq(t - last) + bytes([0x90 if vel else 0x80, note, vel or 64]); last = t
    ev += _vlq(0) + bytes([0xFF, 0x2F, 0x00])
    return bytes(ev)
def smf(path=None, notes=None):
    trk = _track({"kick": KICK, "snare": SNAR, "hat": HAT}, notes)
    blob = b"MThd" + (6).to_bytes(4, "big") + b"\x00\x00\x00\x01" + PPQ.to_bytes(2, "big")
    blob += b"MTrk" + len(trk).to_bytes(4, "big") + trk
    dest = path or Path(__file__).resolve().parent / "vault" / "deck.mid"
    dest.parent.mkdir(exist_ok=True); dest.write_bytes(blob)
    return {"bytes": len(blob), "path": str(dest), "ppq": PPQ}
