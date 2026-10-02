"""Load a local file only. Absent class or missing path denies. No pull."""
from pathlib import Path
def load(path, present):
    file = Path(path)
    if not present:
        return {"ok": False, "why": "no_device", "loaded": False, "model_pull": False}
    if not file.is_file():
        return {"ok": False, "why": "missing", "loaded": False, "model_pull": False}
    return {"ok": True, "why": "local", "loaded": True, "bytes": file.stat().st_size, "model_pull": False}
