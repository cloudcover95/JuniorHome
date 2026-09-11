#!/usr/bin/env python3
"""Sibling workspace. Clone only if missing. Never vendor into Home."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = json.loads((ROOT / "catalog" / "repos.json").read_text(encoding="utf-8"))
OWNER = CAT["owner"]


def base() -> Path:
    env = os.environ.get("JUNIORCLOUD")
    if env:
        return Path(env)
    return Path.home() / "JuniorCloud"


def names() -> list[str]:
    return [r["name"] for r in CAT["repos"] if r["name"] != "Junior-PDF"]


def status() -> list[dict]:
    root = base()
    out = []
    for r in CAT["repos"]:
        p = root / r["name"]
        out.append({**r, "path": str(p), "present": p.is_dir()})
    return out


def pythonpath() -> str:
    root = base()
    return ":".join(str(root / n) for n in CAT["pythonpath"])


def ensure(pull: bool = False) -> None:
    root = base()
    root.mkdir(parents=True, exist_ok=True)
    for r in CAT["repos"]:
        if r["name"] in {"Junior-PDF", "cloudcover95"}:
            continue
        dest = root / r["name"]
        if dest.is_dir():
            if pull:
                subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
            continue
        url = f"https://github.com/{OWNER}/{r['name']}.git"
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=False)


def main(argv: list[str]) -> int:
    if "--ensure" in argv:
        ensure(False)
    if "--pull" in argv:
        ensure(True)
    rows = status()
    missing = [r["name"] for r in rows if not r["present"] and r["name"] not in {"Junior-PDF", "cloudcover95"}]
    print(
        json.dumps(
            {
                "root": str(base()),
                "pythonpath": pythonpath(),
                "present": sum(1 for r in rows if r["present"]),
                "missing": missing,
                "repos": [{"name": r["name"], "role": r["role"], "present": r["present"]} for r in rows],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
