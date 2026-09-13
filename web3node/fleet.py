"""Audited homelab fleet."""
FLEET = {
    "T4_floor": {"hw": "Radxa Zero 3W / Orange Pi Zero 2W", "ram_gb": 4, "watts": 3,
                 "ok": ("trit-ticket", "telemetry"), "no": ("numpy-heavy-vr", "gguf", "ue5"), "kit_usd": 49},
    "T4_ship": {"hw": "Raspberry Pi 5 8GB + cooler + NVMe HAT + PSU", "ram_gb": 8, "watts": 12,
                "ok": ("juniorosai-kernel", "absmean", "csi", "trit-pack"), "no": ("ue5", "70b"), "kit_usd": 180},
    "T4_vision": {"hw": "Jetson Orin Nano 8GB", "ram_gb": 8, "watts": 15,
                  "ok": ("mipi", "int8-tops", "trit-pack"), "no": ("ue5", "70b"), "kit_usd": 249},
    "T0_home": {"hw": "M4 Mac mini 24GB", "ram_gb": 24, "watts": 45,
                "ok": ("mlx", "numpy", "gguf-if-local", "frameforge2d", "blender-cmd"), "no": ("ue5-boot", "70b-board"), "kit_usd": 900},
}

def pick(watts, workload=""):
    t = (workload or "").lower()
    if watts <= 4 or "floor" in t or "telemetry" in t:
        return "T4_floor"
    if "vision" in t or "mipi" in t or "orin" in t:
        return "T4_vision"
    if watts <= 15 or "lidar" in t or "robot" in t or "csi" in t:
        return "T4_ship"
    return "T0_home"

def isolate(payload, watts):
    cls = pick(watts, str(payload.get("workload") or ""))
    row = FLEET[cls]
    return {"class": cls, "hw": row["hw"], "watts": row["watts"], "ue_boot": False,
            "gguf": cls == "T0_home", "payload": "trit-ticket" if cls.startswith("T4") else "kernel+optional-gguf"}
