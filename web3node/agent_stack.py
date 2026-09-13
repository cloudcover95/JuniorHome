"""Ng four patterns: Architect -> TechLead -> Developer + reflection."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from compute_profiles import pick
from fieldcore_bridge import pick_juniorllm_port
from home_terraform import inject

@dataclass
class Handoff:
    role: str
    task: str
    profile: str
    port: str
    notes: list[str]
    reflect: str

def _guard(text: str) -> dict:
    try:
        import importlib
        g = importlib.import_module("agent.guardrails")
        if hasattr(g, "scan_skill_source"):
            rail = g.scan_skill_source(text)
            return {"ok": getattr(rail, "ok", True), "source": "juniorllm"}
    except Exception:
        pass
    banned = ("ignore previous", "0.0.0.0", "curl http")
    return {"ok": not any(b in text.lower() for b in banned), "source": "home-lite"}

def run(task: str) -> dict:
    rail = _guard(task)
    if not rail["ok"]:
        return {"ok": False, "guard": rail}
    profile, port, tf = pick(task), pick_juniorllm_port(task), inject(task)
    handoffs = [
        Handoff("Architect", task, profile, port, ["split offline DEM vs UE5", "Home writes plans"], "Unreal off T0. Spark gets world-model plan."),
        Handoff("TechLead", task, profile, port, ["Omega blender_bridge", "FieldCore intent"], "Tools: blender_bridge, live_book, ph_infer, trit_cache."),
        Handoff("Developer", task, profile, port, ["emit capsule cmd", "no UE on 45W"], "If T0 and task wants UE5, export plan.json only."),
    ]
    out = {"ok": True, "patterns": ["planning", "tool-use", "multi-agent", "reflection"], "guard": rail,
           "terraform": {k: tf.get(k) for k in ("port", "ok", "fusion_backend")},
           "handoffs": [asdict(h) for h in handoffs]}
    path = Path(__file__).resolve().parent / "vault" / "agent_stack.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["path"] = str(path)
    return out

if __name__ == "__main__":
    print(json.dumps(run("offline low power lidar terrain capsule blender omega"), indent=2))
