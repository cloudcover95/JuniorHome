"""Native deck hardware. Class node only. Does not open the device."""
import platform
from pathlib import Path
from write_gate import commit
OUT = Path.home() / ".juniorhome" / "deck" / "native.txt"
AUDIO = ("/dev/snd/pcmC0D0c", "/dev/snd/pcmC0D0p")
HID = ("/dev/hidraw0",)
MIDI = ("/dev/snd/midiC0D0",)
def _hit(paths):
    return any(Path(p).exists() for p in paths)
def probe():
    audio, hid, midi = _hit(AUDIO), _hit(HID), _hit(MIDI)
    return {"sys": platform.system().lower(), "audio": audio, "hid": hid, "midi": midi,
            "present": audio or hid or midi, "opened": False, "live": False}
def admit(env="open"):
    row = probe()
    text = f"{row['sys']} a={int(row['audio'])} h={int(row['hid'])} m={int(row['midi'])}"
    written = commit(env, OUT, text)
    row.update({"env": env, "disk": written["disk"], "why": written["why"], "ticket": text})
    return row
