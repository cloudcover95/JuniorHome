"""JuniorDrive bridge. Flips AGI_SDK 70B claim."""
from adapters import run as adapters_run
from fleet import isolate

PROFILES = {
    "T4_ship": {"precision": "ternary-1.58", "max": "ticket", "hw": "Pi 5 8GB"},
    "T0_m6": {"precision": "ternary-1.58", "max": "kernel+local-gguf", "hw": "M6/M4 mini"},
    "T1_spark": {"precision": "fp8/int8-ok", "max": "fine-tune-latch", "hw": "Spark"},
}

def route(task, watts=12.0):
    node = isolate({"workload": task}, watts)
    key = node["class"]
    return {**PROFILES.get(key, PROFILES["T4_ship"]), "class": key, "ue_boot": node["ue_boot"]}

def sim_to_real():
    pipe = adapters_run(n=3, train_run=False)
    return {"drive": "JuniorDrive", "pipe": "JuniorOSai-adapters", "route": route("robot csi", 12),
            "mean_y": pipe.get("mean_y"), "port": pipe.get("port"), "train": pipe.get("run"),
            "not": ["galkreiser1-juniordrive", "70B-on-mini", "zeRO-3"]}

if __name__ == "__main__":
    import json
    print(json.dumps(sim_to_real(), indent=2))
