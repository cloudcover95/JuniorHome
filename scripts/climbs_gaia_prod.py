#!/usr/bin/env python3
"""JuniorHome probe that talks to Climbs Gaia cores if they are on disk.

Pollinate, do not destroy. Falls back to a local envelope if Climbs is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SIBLING = HOME.parent / "JuniorClimbs"
if SIBLING.is_dir():
    sys.path.insert(0, str(SIBLING))

def main() -> int:
    try:
        from core.gaia_threads import GaiaPool

        pulse = GaiaPool().pulse("home climbs beta", area="home")
        print(json.dumps({"via": "JuniorClimbs.core", "ok": pulse.get("ok"), "witness": pulse.get("witness")}, indent=2))
        return 0 if pulse.get("ok") else 1
    except Exception as exc:
        print(json.dumps({"via": "home-fallback", "ok": True, "error": type(exc).__name__, "download": False}, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
