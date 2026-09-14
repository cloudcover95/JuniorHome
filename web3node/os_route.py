"""JuniorOS / JuniorOSai surface pick. Does not boot UE5."""
from __future__ import annotations


def route(task: str, watts: float = 45.0) -> dict:
    t = (task or "").lower()
    if any(k in t for k in ("ue5", "unreal", "spark")) or watts >= 80:
        surface, profile, launch = "ue5", "T1_spark", False
    elif any(k in t for k in ("frameforge", "ff2d", "2d fight", "canvas")):
        surface, profile, launch = "frameforge2d", "T0_home", False
    elif any(k in t for k in ("flagstaff", "fieldcore", "boulder", "xanadu", "stonefield")):
        surface, profile, launch = "fieldcore-intent", "T4_mobile", False
    elif any(k in t for k in ("juniorosai", "osai", "overlay")):
        surface, profile, launch = "juniorosai", "T0_home", False
    elif any(k in t for k in ("dxf", "cad", "omega")):
        surface, profile, launch = "blender-cmd", "T0_home", False
    else:
        surface, profile, launch = "home-clock", "T0_home", False
    port = "JuniorBitNetDraft" if surface == "blender-cmd" else (
        "JuniorBitNetFieldCore" if surface == "fieldcore-intent" else "JuniorAstra"
    )
    return {
        "task": task,
        "surface": surface,
        "profile": profile,
        "port": port,
        "launch": launch,
        "bind": "127.0.0.1",
        "ue5": surface == "ue5",
        "ff2d": surface == "frameforge2d",
        "os": "JuniorOS" if "os" in t else "Home",
    }


if __name__ == "__main__":
    import json
    for s in (
        "frameforge2d loopback",
        "ue5 world model spark",
        "field flagstaff",
        "juniorosai overlay",
        "omega title block",
        "buy oats",
    ):
        print(json.dumps(route(s, 90 if "ue5" in s else 45)))
