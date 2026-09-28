import shutil
from music_deck import deck
PORTS = (("hydrogen:out_L", "ardour:Hydrogen/audio_in 1"),
         ("hydrogen:out_R", "ardour:Hydrogen/audio_in 2"),
         ("midi_pad:out", "hydrogen:midi_in"),
         ("ardour:master_out L", "system:playback_1"),
         ("ardour:master_out R", "system:playback_2"))
def wire():
    return {"graph": [list(p) for p in PORTS],
            "pipewire": bool(shutil.which("pw-cli") or shutil.which("pipewire")),
            "jack": bool(shutil.which("jackd") or shutil.which("jack_lsp")),
            "connected": False, "bounce": "audacity after export",
            "deck": deck()}
