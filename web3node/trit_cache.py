"""Pack ternary states and compare compression.

stdlib zlib is always available. zstandard is used when installed
(BitNet-mlx / lake hosts). Packed trits beat raw float32 JSON every time.
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

try:
    import zstandard as zstd  # type: ignore

    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False


def pack_trits(matrix: Any) -> bytes:
    """2 bits per trit: -1->00, 0->01, +1->10. Four trits per byte."""
    flat = np.rint(_as_np(matrix)).astype(np.int8).clip(-1, 1).reshape(-1)
    codes = np.where(flat < 0, 0, np.where(flat == 0, 1, 2)).astype(np.uint8)
    pad = (-codes.size) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    packed = codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4) | (codes[3::4] << 6)
    header = np.array([flat.size], dtype=np.uint32).tobytes()
    return header + packed.tobytes()


def zlib_compress(blob: bytes, level: int = 9) -> bytes:
    return zlib.compress(blob, level)


def zstd_compress(blob: bytes, level: int = 10) -> bytes:
    if not HAVE_ZSTD:
        raise RuntimeError("zstandard not installed")
    return zstd.ZstdCompressor(level=level).compress(blob)


def compare(matrix: Any) -> dict[str, Any]:
    raw = _as_np(matrix).astype(np.float32).tobytes()
    packed = pack_trits(matrix)
    row = {
        "float32_bytes": len(raw),
        "trit_pack_bytes": len(packed),
        "zlib_on_pack": len(zlib_compress(packed)),
        "zlib_on_float32": len(zlib_compress(raw)),
        "zstd_available": HAVE_ZSTD,
    }
    if HAVE_ZSTD:
        row["zstd_on_pack"] = len(zstd_compress(packed))
        row["zstd_on_float32"] = len(zstd_compress(raw))
        row["winner"] = min(
            ("zlib_on_pack", row["zlib_on_pack"]),
            ("zstd_on_pack", row["zstd_on_pack"]),
            key=lambda item: item[1],
        )[0]
    else:
        row["winner"] = "zlib_on_pack"
        row["note"] = "zstandard missing — zlib on packed trits is the portable winner"
    return row


def _as_np(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "tolist"):
        return np.array(matrix.tolist(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)
