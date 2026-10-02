"""Production status. Planned slots, no device ran."""
import shutil
def status():
    return {"planned": {"pi5": 4, "gpu": 5, "spark": 1, "m5-pro": 1},
            "attached": 0, "cuda": bool(shutil.which("nvidia-smi")),
            "amx": False, "mps": False, "ran": False, "model_pull": False}
