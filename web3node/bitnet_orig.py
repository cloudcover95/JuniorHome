"""Original-Python AbsMean / sign / I2S — same math as JuniorLLM adaptations/omega_cad/quant.py."""
from __future__ import annotations


def absmean(xs: list[float]) -> tuple[list[int], float]:
    if not xs:
        return [], 1.0
    delta = sum(abs(x) for x in xs) / len(xs) or 1.0
    out = []
    for x in xs:
        q = round(x / delta)
        out.append(1 if q > 1 else (-1 if q < -1 else int(q)))
    return out, delta


def sign(xs: list[float]) -> list[int]:
    return [1 if x > 0 else (-1 if x < 0 else 0) for x in xs]


def i2s_pack(trits: list[int]) -> bytes:
    acc = 0
    n = 0
    out = bytearray()
    for t in trits:
        c = 0 if t == 0 else (1 if t == 1 else 2)
        acc = (acc << 2) | c
        n += 2
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)
