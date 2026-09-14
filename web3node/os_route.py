"""JuniorOS / JuniorOSai surface pick. Does not boot UE5."""
from __future__ import annotations


def route(task: str, watts: float = 45.0) -> dict:
    t = (task or "").lower()
    if any(k in t for k in ("gaia", "companion", "portrait", "goldend")):
        surface, profile, launch, port = "gaia-portrait", "T0_home", False, "JuniorGaia"
    elif any(k in t for k in ("ue5", "unreal", "spark")) or watts >= 80:
        surface, profile, launch, port = "ue5", "T1_spark", False, "JuniorAstra"
    elif any(k in t for k in ("frameforge", "ff2d", "2d fight", "canvas")):
        surface, profile, launch, port = "frameforge2d", "T0_home", False, "JuniorAstra"
    elif any(k in t for k in ("flagstaff", "fieldcore", "boulder", "xanadu", "stonefield")):
        surface, profile, launch, port = "fieldcore-intent", "T4_mobile", False, "JuniorBitNetFieldCore"
    elif any(k in t for k in ("juniorosai", "osai", "overlay")):
        surface, profile, launch, port = "juniorosai", "T0_home", False, "JuniorAstra"
    elif any(k in t for k in ("dxf", "cad", "omega")):
        surface, profile, launch, port = "blender-cmd", "T0_home", False, "JuniorBitNetDraft"
    else:
        surface, profile, launch, port = "home-clock", "T0_home", False, "JuniorAstra"
    return {
        "task": task,
        "surface": surface,
        "profile": profile,
        "port": port,
        "launch": launch,
        "bind": "127.0.0.1",
        "ue5": surface == "ue5",
        "ff2d": surface == "frameforge2d",
        "gaia": surface == "gaia-portrait",
        "omega_mesh": "stub" if surface == "gaia-portrait" else None,
        "os": "JuniorOS" if "os" in t else "Home",
    }


if __name__ == "__main__":
    import json
    for s in (
        "gaia they home",
        "frameforge2d loopback",
        "ue5 world model spark",
        "field flagstaff",
        "omega title block",
    ):
        print(json.dumps(route(s, 90 if "ue5" in s else 45)))
