import json, shutil
from pathlib import Path
def profile():
    layout = json.loads((Path(__file__).resolve().parent/"geode_deck.json").read_text(encoding="utf-8"))
    layout["blender"] = bool(shutil.which("blender"))
    layout["trit_mcu"] = False
    layout["ram"] = {"T4_GB": 8, "mcu_kb": 264, "T0": "host"}
    layout["live"] = False
    return layout
