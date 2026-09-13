"""Handheld IoT pack. JuniorDrive is a real tree."""
from __future__ import annotations
import json
from pathlib import Path
from dem_tile import ingest
from fieldcore_bridge import pick_juniorllm_port
from home_kernel import dispatch
from home_terraform import inject
from omega_dryrun import dry_run
from second_brain import write_note
TREES = {
    "JuniorDrive": "https://github.com/cloudcover95/JuniorDrive",
    "JuniorClimbs/StoneField": "https://github.com/cloudcover95/JuniorClimbs",
    "JuniorOmega": "https://github.com/cloudcover95/JuniorOmega",
    "AGI_SDK": "https://github.com/cloudcover95/AGI_SDK",
    "JuniorLLM/Flagstaff": "https://github.com/cloudcover95/JuniorLLM",
    "JuniorMemSys-Suite": "https://github.com/cloudcover95/JuniorMemSys-Suite",
    "stocksnode": "https://github.com/cloudcover95/stocksnode",
}
def pack():
    kernel = dispatch("handheld iot dem trit", "terrain", watts=12)
    tile = ingest()
    cap = dry_run("omega-lidar")
    tf = inject("flagstaff field handheld drive stonefield")
    note = write_note(Path(__file__).resolve().parent / "vault", "# handheld iot\n")
    return {"kernel": kernel, "port": pick_juniorllm_port("flagstaff field handheld"),
            "terraform": {k: tf.get(k) for k in ("port", "ok", "fusion_backend")},
            "dem": {k: tile.get(k) for k in ("site", "z_mean", "z_min", "z_max", "misses", "step_m", "ply")},
            "omega": {k: cap.get(k) for k in ("kind", "status", "host", "source")},
            "obsidian": str(note), "trees": TREES}
if __name__ == "__main__":
    print(json.dumps(pack(), indent=2, default=str))
