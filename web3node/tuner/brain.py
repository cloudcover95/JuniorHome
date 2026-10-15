"""Brain note. Local JSONL. No model. No flash."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def note(text: str, close: str = "") -> dict[str, Any]:
    return {"text": text, "close": close, "kind": "tuner_brain", "flash": False}

def write(text: str, close: str = "") -> Path:
    out = Path.home() / ".juniorhome" / "brain"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "tuner.jsonl"
    p.open("a", encoding="utf-8").write(json.dumps(note(text, close)) + "\n")
    return p

if __name__ == "__main__":
    print(write("run complete", "deadbeef"))
