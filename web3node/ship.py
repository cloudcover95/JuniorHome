from ep_hooks import hooks
from iot_runtime import pick
from spark_domain import domain
from weekly_bench import weekly
def card():
    w = weekly()
    return {"product": "JuniorOS", "model": "JuniorOSai", "ship": "prototype",
            "checkpoint_2b": False, "runtime": pick()["runtime"],
            "weekly_ok": w["sandbox_ok"] and w["agents"], "fuse_us": w["fuse_us"],
            "hooks_default": hooks()["default"], "t4": "fused-list + trit5"}
