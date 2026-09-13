"""StoneField Home module. Climbs gym / spatial intent on FieldCore.
Does not write POS. Wear/grade are scores.
"""
from __future__ import annotations

GRADES = ("5.10d", "5.11c", "5.12b", "V7", "V9", "V3")
WEAR_FLAG = 0.79


def grade_from_tag(tag_hash):
    idx = abs(int(tag_hash)) % len(GRADES)
    return {"grade": GRADES[idx], "index": idx, "confidence": 0.91}


def wear_action(hold_wear):
    return "FLAG_MAINTENANCE" if hold_wear > WEAR_FLAG else "nominal"


def features_from_gym(hold_wear, traffic, lidar_density):
    flag = 1.0 if hold_wear > WEAR_FLAG else 0.0
    return [lidar_density, 1.0 - hold_wear, traffic, 0.0, flag, 0.0, 0.0, 0.0, 0.35, hold_wear, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def tick(hold_wear=0.4, traffic=0.2, lidar_density=0.6):
    import importlib
    feat = features_from_gym(hold_wear, traffic, lidar_density)
    envelope = {"port": "JuniorBitNetFieldCore", "trit": 0}
    action = "hold"
    for name in ("packs.fieldcore.sandbox", "sandbox"):
        try:
            sb = importlib.import_module(name)
            out = sb.Sandbox().eval(feat)
            envelope = out["envelope"]
            action = out["action"]
            break
        except Exception:
            continue
    else:
        action = "flag" if hold_wear > WEAR_FLAG else "hold"
    return {
        "action": action,
        "wear": wear_action(hold_wear),
        "grade_hint": grade_from_tag(int(hold_wear * 1000)),
        "envelope": envelope,
        "tree": "JuniorClimbs",
        "module": "stonefield",
    }


def self_test():
    ok = tick(0.4)
    hot = tick(0.88)
    assert ok["wear"] == "nominal"
    assert hot["wear"] == "FLAG_MAINTENANCE"
    return {"ok": True, "cool": ok["action"], "hot_wear": hot["wear"]}
