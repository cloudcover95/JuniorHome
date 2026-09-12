from __future__ import annotations
import unittest
from .. import backends, quant
FF_IN = [-2.0, -0.01, 0.0, 0.02, 2.0]
FF_OUT = [-1, 0, 0, 0, 1]
class TestBackends(unittest.TestCase):
    def test_frameforge_absmean_vector(self):
        trits, scale = quant.quant_vec(FF_IN)
        self.assertEqual(trits, FF_OUT)
        self.assertGreater(scale, 0)
    def test_all_domains_match_stdlib(self):
        gold, _, _ = backends.absmean_quantize(FF_IN, backend="stdlib")
        self.assertEqual(gold, FF_OUT)
        for name in backends.DOMAINS:
            got, _, used = backends.absmean_quantize(FF_IN, backend=name)
            self.assertEqual(got, gold, msg="%s vs stdlib (%s)" % (name, used))
    def test_detect_keys(self):
        self.assertIn(backends.detect()["preferred"], backends.DOMAINS)
if __name__ == "__main__":
    unittest.main()
