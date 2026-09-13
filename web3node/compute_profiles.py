PROFILES = {
    "T0_home": {"name": "Home clock (~45W, MLX or numpy)", "ok": ("mlx", "numpy", "blender-cmd", "frameforge2d"), "no": ("unreal-boot",)},
    "T1_spark": {"name": "RTX Spark N1X / DGX Spark", "ok": ("cuda", "ue5", "blender", "world-model-small"), "no": ("taalas-retarget",)},
    "T2_discrete": {"name": "RTX 50-class PCIe", "ok": ("ue5", "lingbot-class"), "no": ()},
    "T3_asic": {"name": "Taalas-class hard-coded inference", "ok": ("frozen-trit-ticket",), "no": ("weight-update",)},
    "T4_mobile": {"name": "phone NPU / Pi / Jetson", "ok": ("trit-pack", "fieldcore-intent", "dem-slice"), "no": ("ue5",)},
}

def pick(task: str, watts: float = 45.0) -> str:
    t = (task or "").lower()
    if "asic" in t or "taalas" in t or "silicon" in t:
        return "T3_asic"
    if "unreal" in t or "ue5" in t or "spark" in t:
        return "T1_spark" if watts >= 60 else "T0_home"
    if "mobile" in t or "offline" in t or "lidar" in t or "dem" in t:
        return "T4_mobile" if watts < 20 else "T0_home"
    return "T0_home"
