import json
from pathlib import Path
from beat_pack import pack
from music_map import apply
from music_smf import smf
def bundle(name="junior"):
    root = Path(__file__).resolve().parent / "vault" / "deck_bundle"
    root.mkdir(parents=True, exist_ok=True)
    raw = pack()
    (root / "beat.jdb1").write_bytes(raw)
    apply(name)
    mid = smf(path=root / "deck.mid")
    man = {"engine": "JuniorDeck", "map": name, "jdb1_B": len(raw),
           "smf_B": mid["bytes"], "live": False}
    (root / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    man["dir"] = str(root)
    return man
