"""Trit stays on the local envelope device. Host RAM is not the engine."""
import shutil
LAYERS = {"t4": "gpu", "t0": "mlx"}
def device(layer="t4"):
    want = LAYERS.get(layer, "gpu")
    gpu = bool(shutil.which("nvidia-smi"))
    try:
        import mlx.core
        mlx = True
    except Exception:
        mlx = False
    present = gpu if want == "gpu" else mlx
    return {"layer": layer if layer in LAYERS else "t4", "want": want, "present": present,
            "ram_engine": False, "disk": False, "admit": present, "model_pull": False}
