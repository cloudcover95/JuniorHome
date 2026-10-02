"""Local backends. Named is not present."""
import shutil
NAMED = ("sram", "amx", "mps", "cuda")
def backends():
    cuda = bool(shutil.which("nvidia-smi"))
    try:
        import mlx.core
        mlx = True
    except Exception:
        mlx = False
    return {"named": list(NAMED), "cuda": cuda, "amx": False, "mps": False,
            "sram": False, "mlx": mlx, "ran": False, "model_pull": False}
