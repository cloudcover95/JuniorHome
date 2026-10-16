"""Native Home probe. JuniorOS ships Home. Local only."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def probe() -> dict[str, Any]:
    home = Path.home() / "JuniorHome"
    os_dir = Path.home() / ".juniorhome" / "os"
    body = {
        "home_present": home.is_dir(),
        "os_dir": str(os_dir),
        "native": True,
        "local": True,
    }
    out = os_dir / "native_home.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body

if __name__ == "__main__":
    print(json.dumps(probe(), indent=2))
