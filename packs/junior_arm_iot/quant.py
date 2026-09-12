"""AbsMean ternary. Matches BitNet-mlx + FrameForge bitnet_quant.

W_q = clip(round(W / mean(|W|)), -1, 1)  ==  sign(W) if |W|/scale > 0.5
BitNet scores CPU intent only. Not a Nintendo product.
"""
from __future__ import annotations


def absmean(xs):
    if not xs:
        return 0.0
    return sum(abs(x) for x in xs) / len(xs)


def quant_trit(x, scale):
    if scale <= 1e-12:
        return 0
    v = x / scale
    if v > 0.5:
        return 1
    if v < -0.5:
        return -1
    return 0


def quant_vec(xs):
    scale = absmean(xs)
    return [quant_trit(x, scale) for x in xs], scale


def pack_trits(trits):
    out = bytearray()
    n = len(trits)
    i = 0
    while i < n:
        acc = 0
        mul = 1
        for _ in range(5):
            t = trits[i] if i < n else 0
            acc += (t + 1) * mul
            mul *= 3
            i += 1
        out.append(acc & 255)
    return bytes(out)


def unpack_trits(buf, n):
    out = []
    for b in buf:
        v = b
        for _ in range(5):
            if len(out) >= n:
                return out
            out.append((v % 3) - 1)
            v //= 3
    return out[:n]


def bitlinear(x, w, scale):
    acc = 0.0
    n = min(len(x), len(w))
    for i in range(n):
        t = w[i]
        if t:
            acc += x[i] if t > 0 else -x[i]
    return acc * scale


def mmio_features(lx, ly, btns, uart_n, gpio, fwd, cycles):
    c = float(max(1, cycles))
    return [lx / 127.0, ly / 127.0, (btns & 255) / 255.0, uart_n / 64.0, (gpio & 255) / 255.0, fwd / c, 1.0 if btns else 0.0, min(1.0, c / 4096.0)]
