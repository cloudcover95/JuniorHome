"""Hardware profiles from community stacks. Local channel map. No flash."""
from __future__ import annotations
import json
from typing import Any

PROFILES = {
    "speeduino": {
        "source": "Speeduino / LibreTune INI",
        "channels": {"rpm": "RPM", "afr": "AFR", "tps": "TPS", "map": "MAP", "ect": "CLT", "iat": "IAT", "vss": "VSS"},
        "flash": False,
    },
    "opendyno": {
        "source": "OpenDyno GPS road",
        "channels": {"vss": "speed_mps", "rpm": "rpm"},
        "flash": False,
    },
    "diydyno": {
        "source": "DIY-Dyno / DeeEmm load cell + tach",
        "channels": {"rpm": "tach", "force_n": "load_cell"},
        "flash": False,
    },
    "hondash": {
        "source": "HonDash K-Pro / S300",
        "channels": {"rpm": "RPM", "afr": "AFR", "tps": "TPS", "map": "MAP", "ect": "ECT", "iat": "IAT", "vss": "VSS"},
        "flash": False,
    },
}

def map_sample(profile: str, raw: dict[str, Any]) -> dict[str, Any]:
    p = PROFILES.get(profile)
    if not p:
        return {"ok": False, "reason": "unknown_profile"}
    out = {}
    for ours, theirs in p["channels"].items():
        if theirs in raw:
            out[ours] = raw[theirs]
    out["profile"] = profile
    out["flash"] = False
    return {"ok": True, "sample": out}

if __name__ == "__main__":
    print(json.dumps(map_sample("speeduino", {"RPM": 4500, "AFR": 14.7, "CLT": 90}), indent=2))
