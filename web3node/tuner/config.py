"""Local config. Rate 0.05. No flash."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DEFAULT = {"rate": 0.05, "authority": 0.05, "flash": False, "profiles": ["speeduino", "opendyno", "diydyno", "hondash"]}

def load(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else Path.home() / ".juniorhome" / "os" / "tuner.toml"
    if not p.is_file():
        return DEFAULT.copy()
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return DEFAULT.copy()

def write(cfg: dict[str, Any] | None = None) -> Path:
    out = Path.home() / ".juniorhome" / "os" / "tuner.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg or DEFAULT, indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    print(write())
