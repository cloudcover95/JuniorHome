"""Compile live plate metrics. No USGS fetch. No UE5 launch."""
from __future__ import annotations

import json
import math
import time
import zlib
from pathlib import Path


def _hf(n: int, seed: int = 42) -> list[list[float]]:
    grid = []
    for j in range(n):
        row = []
        for i in range(n):
            u, v = i / max(1, n - 1), j / max(1, n - 1)
            row.append(math.sin((u * 7 + seed % 9) * 1.7) * 0.35 + math.cos((v * 5) * 1.3) * 0.25)
        grid.append(row)
    return grid


def _trit(xs: list[float]) -> tuple[list[int], float]:
    abs_w = [abs(x) for x in xs]
    gamma = sum(abs_w) / max(1, len(abs_w)) + 1e-7
    return [max(-1, min(1, int(round(x / gamma)))) for x in xs], gamma


def compile_plate(n: int = 32) -> dict:
    t0 = time.perf_counter()
    g = _hf(n)
    flat = [z for row in g for z in row]
    trit, gamma = _trit(flat)
    raw = b"".join(int(t + 1).to_bytes(1, "little") for t in trit)
    packed = zlib.compress(raw, 9)
    f32 = n * n * 4
    ms = (time.perf_counter() - t0) * 1000
    return {
        "n": n,
        "verts": n * n,
        "faces": (n - 1) ** 2,
        "gamma": round(gamma, 4),
        "sparsity": round(sum(1 for t in trit if t == 0) / len(trit), 3),
        "float32_b": f32,
        "trit_zlib_b": len(packed),
        "ratio": round(f32 / max(1, len(packed)), 2),
        "gen_ms": round(ms, 3),
        "ue5_launch": False,
        "usgs_fetch": False,
    }


def write_train(out: Path, n: int = 32) -> Path:
    g = _hf(n)
    trit, _ = _trit([z for row in g for z in row])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        k = 0
        for j, row in enumerate(g):
            for i, z in enumerate(row):
                fh.write(json.dumps({"x": i, "y": z, "z": j, "trit": trit[k], "site": "gaia-synth"}) + "\n")
                k += 1
    return out


if __name__ == "__main__":
    vault = Path.home() / ".juniorhome" / "gaia_mesh"
    plate = compile_plate(32)
    train = write_train(vault / "gaia_train.jsonl", 32)
    plate["train"] = str(train)
    print(json.dumps(plate, indent=2))
