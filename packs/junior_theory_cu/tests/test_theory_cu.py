from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from packs.junior_theory_cu.fund import Fund


class TheoryCuTests(unittest.TestCase):
    def test_join_propose_vote_board(self):
        with tempfile.TemporaryDirectory() as td:
            f = Fund(Path(td))
            f.join("nico")
            f.join("ada")
            p = f.propose("nico", "tool library")
            f.vote("nico", p.pid, "yes")
            f.vote("ada", p.pid, "yes")
            self.assertEqual(f.treasury, 2)
            self.assertGreaterEqual(len(f.forum.events), 4)
            self.assertTrue((Path(td) / "net" / "units.jsonl").exists())
            b = f.forum.export_bundle()
            self.assertEqual(b["format"], "junior-gossip-v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
