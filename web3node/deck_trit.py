"""Deck trit ticket. Envelope device or nothing."""
import shutil
ENVS = {"t4": "gpu", "t0": "mlx"}
def pack(n=16):
    xs = [1.0 if i % 4 == 0 else 0.0 for i in range(n)]
    gamma = sum(abs(x) for x in xs) / len(xs) or 1.0
    trits = []
    for x in xs:
        q = round(x / gamma)
        trits.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return trits, gamma
def ticket(env="t4"):
    want = ENVS.get(env, "gpu")
    gpu = bool(shutil.which("nvidia-smi"))
    try:
        import mlx.core
        mlx = True
    except Exception:
        mlx = False
    present = gpu if want == "gpu" else mlx
    trits, gamma = pack()
    return {"env": env if env in ENVS else "t4", "want": want, "admit": present,
            "n": len(trits), "zeros": trits.count(0), "gamma": round(gamma, 3),
            "ram_engine": False, "disk": False, "model_pull": False}
