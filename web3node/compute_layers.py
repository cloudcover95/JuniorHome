"""Infer / train-data / train-dev layers. Solar feeds T4 watts."""
from __future__ import annotations
import json
from pathlib import Path
from fleet import isolate
from home_terraform import inject

LAYERS = {
    "infer_edge": {"watts": 12, "workload": "robot csi infer", "train": False, "solar": True},
    "infer_home": {"watts": 45, "workload": "m6 home infer", "train": False, "solar": False},
    "train_data": {"watts": 12, "workload": "robot telemetry dataset", "train": False, "solar": True},
    "train_dev": {"watts": 90, "workload": "ue5 spark fine-tune", "train": True, "solar": False},
}

def expand(ask="juniorosai field infer train"):
    rows = {}
    for name, spec in LAYERS.items():
        node = isolate({"workload": spec["workload"]}, spec["watts"])
        tf = inject(f"{ask} {name}")
        rows[name] = {**spec, "node": node["class"], "hw": node["hw"],
                      "ue_boot": node["ue_boot"], "port": tf.get("port"), "tf_ok": tf.get("ok")}
    out = {"solar": "free T4 watts if the panel feeds the Pi",
           "train_policy": "T4 collects. T0 labels. T1 may fine-tune. No 70B pretrain.",
           "layers": rows}
    path = Path(__file__).resolve().parent / "vault" / "compute_layers.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    out["out"] = str(path)
    return out

if __name__ == "__main__":
    print(json.dumps(expand(), indent=2, default=str))
