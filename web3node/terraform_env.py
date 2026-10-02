"""Home terraform envelope. T4 GPU, T0 MLX."""
import shutil
ENVS = {"t4": "gpu", "t0": "mlx"}
def envelope(name="t4"):
    want = ENVS.get(name, "gpu")
    gpu = bool(shutil.which("nvidia-smi"))
    try:
        import mlx.core
        mlx = True
    except Exception:
        mlx = False
    present = gpu if want == "gpu" else mlx
    return {"env": name if name in ENVS else "t4", "want": want, "present": present,
            "admit": present, "ram_engine": False, "model_pull": False}
