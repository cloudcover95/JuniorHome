"""Audited homelab fleet."""
FLEET = {
    "T4_floor": {"hw": "Radxa Zero 3W / Orange Pi Zero 2W", "ram_gb": 4, "watts": 3, "kit_usd": 49},
    "T4_ship": {"hw": "Raspberry Pi 5 8GB kit", "ram_gb": 8, "watts": 12, "kit_usd": 180},
    "T4_vision": {"hw": "Jetson Orin Nano 8GB", "ram_gb": 8, "watts": 15, "kit_usd": 249},
    "T0_m4": {"hw": "M4 Mac mini 24GB used", "ram_gb": 24, "watts": 45, "kit_usd": 1000},
    "T0_m6": {"hw": "M6 Mac mini 24GB/512GB", "ram_gb": 24, "watts": 45, "kit_usd": 1299,
              "asin": "B0HGMK9TK3", "url": "https://www.amazon.com/dp/B0HGMK9TK3",
              "ships": "2026-09-22", "applecare3": 1414},
    "T0_m5pro_mbp": {"hw": "M5 Pro MacBook Pro travel", "ram_gb": 24, "watts": 45, "kit_usd": None},
    "T1_spark": {"hw": "RTX Spark / DGX Spark", "ram_gb": 128, "watts": 90, "kit_usd": None},
}

def pick(watts, workload=""):
    t = (workload or "").lower()
    if watts <= 4 or "floor" in t:
        return "T4_floor"
    if "orin" in t or "mipi" in t or "vision" in t:
        return "T4_vision"
    if watts <= 15 or "lidar" in t or "robot" in t or "csi" in t:
        return "T4_ship"
    if watts >= 60 or "spark" in t or "ue5" in t:
        return "T1_spark"
    if "m4" in t:
        return "T0_m4"
    return "T0_m6"

def isolate(payload, watts):
    cls = pick(watts, str(payload.get("workload") or ""))
    row = FLEET[cls]
    return {"class": cls, "hw": row["hw"], "watts": row["watts"],
            "ue_boot": cls == "T1_spark", "gguf": cls.startswith("T0"),
            "payload": "trit-ticket" if cls.startswith("T4") else "kernel+optional-gguf"}
