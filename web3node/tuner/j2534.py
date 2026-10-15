"""J2534 read-only stub. No write. No flash. Local file or empty."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def read_log(path: str | None = None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {"ok": False, "reason": "no_file", "channels": {}}
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "reason": "bad_json", "channels": {}}
    return {"ok": True, "channels": data, "write": False, "flash": False}

if __name__ == "__main__":
    print(json.dumps(read_log(None), indent=2))
