from beat_pack import pack
from deck_mcu import scan
from music_deck import probe
from music_smf import smf
def cycle(pressed=None):
    ticket = scan(pressed)
    return {"receive": {"midi": False, "notes": ticket["down"]},
            "send": {"jdb1_B": len(pack()), "smf_B": smf()["bytes"]},
            "hosts": probe(), "live": False, "update": "pull vault/deck_bundle"}
