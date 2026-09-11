#!/usr/bin/env python3
"""Run the JuniorLLM stack into this Home vault."""
from __future__ import annotations

import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
VAULT = HOME / "vault"
LLM = Path.home() / "JuniorCloud" / "JuniorLLM"
for p in (Path("../JuniorLLM").resolve(), LLM):
    if (p / "scripts" / "all_prod.py").is_file():
        sys.path.insert(0, str(p))
        break
else:
    print("JuniorLLM not next to Home; set PYTHONPATH", file=sys.stderr)
    raise SystemExit(2)

from scripts.all_prod import main as all_prod


def main() -> int:
    VAULT.mkdir(parents=True, exist_ok=True)
    return all_prod(["home_prod", str(VAULT)])


if __name__ == "__main__":
    raise SystemExit(main())
