"""Dash tile: read-only view of latest ~/.juniorhome/obsidian/findings/*.md (from web3node/team_report.py)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FINDINGS = Path.home() / ".juniorhome" / "obsidian" / "findings"
MAX_CHARS = 4000

def tile(path: Path = FINDINGS, max_chars: int = MAX_CHARS) -> dict[str, Any]:
    base: dict[str, Any] = {"tile": "findings", "dir": str(path)}
    try:
        files = [p for p in path.glob("*.md") if p.is_file()] if path.is_dir() else []
        if not files:
            return {**base, "status": "missing", "finding": None}
        latest = max(files, key=lambda p: p.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime, timezone.utc).isoformat()
        text = latest.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {**base, "status": "unreadable", "error": str(e), "finding": None}
    truncated = len(text) > max_chars
    return {**base, "status": "ok", "finding": {
        "path": str(latest), "name": latest.name, "mtime": mtime,
        "text": text[:max_chars], "truncated": truncated}}

if __name__ == "__main__":
    print(json.dumps(tile(), indent=2))
