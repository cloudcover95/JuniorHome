from __future__ import annotations
import unittest
from .. import quant

class TestQuant(unittest.TestCase):
    def test_roundtrip_pack(self):
        src = [1, 0, -1, 1, 1, 0, -1]
        self.assertEqual(quant.unpack_trits(quant.pack_trits(src), len(src)), src)
    def test_absmean_zero(self):
        trits, scale = quant.quant_vec([0.0, 0.0, 0.0])
        self.assertEqual(trits, [0, 0, 0])
        self.assertEqual(scale, 0.0)
    def test_bitlinear_skip_zero(self):
        y = quant.bitlinear([2.0, 3.0, 4.0], [1, 0, -1], 0.5)
        self.assertAlmostEqual(y, (2.0 - 4.0) * 0.5)

if __name__ == "__main__":
    unittest.main()
