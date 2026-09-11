from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEED = {
    "JuniorHome",
    "JuniorLLM",
    "BitNet-mlx",
    "JuniorMemSys-Suite",
    "JuniorEngrTools",
    "JuniorOmega",
    "JuniorStock",
    "JuniorClimbs",
    "crispy-mouse",
}


class CatalogTests(unittest.TestCase):
    def test_titles(self):
        cat = json.loads((ROOT / "catalog" / "repos.json").read_text(encoding="utf-8"))
        names = {r["name"] for r in cat["repos"]}
        self.assertTrue(NEED <= names, NEED - names)
        self.assertIn("JuniorLLM", cat["pythonpath"])
        self.assertIn("JuniorHome", cat["core"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
