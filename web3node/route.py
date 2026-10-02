"""Route a job to a class. No shared memory."""
ROUTE = {"ticket": "pi", "infer": "gpu", "ue": "spark", "mlx": "mac"}
def route(job, present):
    cls = ROUTE.get(job)
    if cls is None:
        return {"ok": False, "why": "unknown", "model_pull": False}
    return {"ok": bool(present.get(cls)), "job": job, "class": cls,
            "shared_memory": False, "model_pull": False}
