"""Omega → OBJ. Blender bpy optional. UE5 off."""
from __future__ import annotations

from pathlib import Path

from ports.gaia_proto import handshake
from ports.trit_gates import clip

JOB = "terrain-obj"


def _obj(path: Path, n: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = [(i % n, i // n, clip(i % 3 - 1) * 0.25) for i in range(n * n)]
    faces = []
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x + 1
            faces.append((a, a + 1, a + n + 1, a + n))
    lines = ["# juniorcloud omega obj", "o gaia_patch"]
    for x, y, z in verts:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for f in faces:
        lines.append("f " + " ".join(str(i) for i in f))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def bpy_present() -> bool:
    try:
        import bpy  # noqa: F401
        return True
    except Exception:
        return False


def harness(note: str = "home dash terrain", out: str | None = None) -> dict:
    env = handshake(note, job=JOB)
    dest = Path(out) if out else Path.home() / ".juniorhome" / "omega" / "patch.obj"
    wrote = _obj(dest) if env.get("schema_ok") else None
    return {
        "schema_ok": env.get("schema_ok"),
        "job": JOB,
        "obj": str(wrote) if wrote else None,
        "bpy": bpy_present(),
        "ue5_launch": False,
        "download": False,
        "svd": False,
    }
