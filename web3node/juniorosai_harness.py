"""JuniorOSai harness. Never fetch weights."""
from __future__ import annotations
import json
from pathlib import Path
from compute_profiles import pick
from home_kernel import dispatch
from prototype_engine import engine
from ternary_kernel import kernel_list, pick_path

MODELS = Path.home() / ".juniorllm" / "models"
GGUF = "bitnet-b1.58-2B-4T-I2_S.gguf"

def probe_disk():
    path = MODELS / GGUF
    present = path.is_file()
    return {"present": present, "path": str(path) if present else None, "fetch": False,
            "fallback": "JuniorOSai-kernel"}

def harness(ask="juniorosai field bitnet", watts=45.0):
    disk = probe_disk()
    profile = pick("ue5 spark" if watts >= 60 else ask, watts)
    row = {"model": "JuniorOSai", "quant": "ternary-1.58",
           "mode": "gguf-i2s" if disk["present"] else "kernel-ternary-1.58",
           "trained_weights_on_disk": disk["present"], "disk": disk, "profile": profile,
           "spark": profile == "T1_spark", "clock": pick_path(), "engine": engine(ask),
           "kernel": kernel_list([0.15, -0.2, 0.05, 0.1], [0.3, -0.25, 0.1, 0.0]),
           "ue_boot": dispatch(ask, domain="llm", watts=watts).get("ue_boot"),
           "open_source": True, "download": False}
    out = Path(__file__).resolve().parent / "vault" / "juniorosai_harness.json"
    out.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    row["out"] = str(out)
    return row

if __name__ == "__main__":
    print(json.dumps(harness(), indent=2, default=str))
