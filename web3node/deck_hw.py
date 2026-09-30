import json, shutil
from pathlib import Path
def profile(rows=None, cols=None):
    layout = json.loads((Path(__file__).resolve().parent/"geode_deck.json").read_text(encoding="utf-8"))
    if rows: layout["keys"]["rows"] = rows
    if cols: layout["keys"]["cols"] = cols
    k = layout["keys"]
    k["count"] = k["rows"] * k["cols"]
    layout["blender"] = bool(shutil.which("blender"))
    layout["trit_mcu"] = False
    layout["ram"] = {"T4_GB": 8, "mcu_kb": 264, "T0": "host"}
    layout["live"] = False
    return layout
