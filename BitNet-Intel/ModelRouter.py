# BitNet-Intel ModelRouter — lists JuniorLLM ports. Compute stays JuniorLLM.

PORTS = [
    {"name": "JuniorBitNetFieldCore", "quant": "ternary-1.58", "download_gb": 0.0},
    {"name": "JuniorAstra", "quant": "n/a", "download_gb": 0.0},
    {"name": "JuniorBitNetDraft", "quant": "ternary-1.58", "download_gb": 0.0},
    {"name": "JuniorGaia", "quant": "ternary-1.58", "download_gb": 0.0, "mesh": "omega-stub"},
    {"name": "JuniorFable", "quant": "n/a", "download_gb": 0.0},
]


class ModelRouter:
    def __init__(self):
        self.hardware_profiles = {
            "apple_silicon": {"preferred_precision": "ternary", "max_model_size": "70B"},
            "jetson": {"preferred_precision": "int4", "max_model_size": "30B"},
            "pi5": {"preferred_precision": "int4", "max_model_size": "7B"},
        }

    def list_ports(self):
        return list(PORTS)

    def route(self, task_type, hardware="apple_silicon"):
        t = (task_type or "").lower()
        profile = self.hardware_profiles.get(hardware, self.hardware_profiles["apple_silicon"])
        name = "JuniorGaia" if any(k in t for k in ("gaia", "companion", "portrait")) else "JuniorAstra"
        return {"port": name, "profile": profile, "ue5_launch": False, "omega_mesh": "stub"}
