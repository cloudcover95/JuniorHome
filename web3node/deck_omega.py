import json
from pathlib import Path
from deck_hw import profile
def job():
    hw = profile()
    spec = {"who": {"mesh": "cad/deck_well.scad", "layout": "web3node/geode_deck.json"},
            "omega": {"repo": "cloudcover95/JuniorOmega", "worker": "blender/headless_worker.py", "launch": False},
            "controls": {"pads": hw["pads"], "knobs": [k["id"] for k in hw["knobs"]],
                         "keys": hw["keys"]["count"], "switch": "MX-hotswap"},
            "compute": hw["ram"], "trit_mcu": False, "blender": hw["blender"]}
    dest = Path(__file__).resolve().parent.parent / "cad" / "omega_job.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec["path"] = str(dest)
    return spec
