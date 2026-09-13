"""FieldCore 1.58 ternary sandbox. Stdlib. JuniorHome.

Envelope, not a second physics and not a gym POS.
Quant backends are declared. Only absmean-trit runs in-process.
mistral.rs Q4K / HQQ / AFQ stay sidecar tags until a GGUF binary exists.
"""
from __future__ import annotations

BITS_PER_TRIT = 1.58496
BACKENDS = {
    "bitnet_158": {"bits": BITS_PER_TRIT, "device": "cpu", "mul": False, "home": True},
    "absmean_trit": {"bits": BITS_PER_TRIT, "device": "cpu", "mul": False, "home": True},
    "q4k": {"bits": 4.5, "device": "cpu|metal|cuda", "mul": True, "home": False, "via": "mistral.rs ISQ"},
    "q8_0": {"bits": 8.0, "device": "cpu", "mul": True, "home": False, "via": "gguf"},
    "hqq4": {"bits": 4.0, "device": "metal|cuda", "mul": True, "home": False, "via": "mistral.rs HQQ"},
    "afq4": {"bits": 4.0, "device": "metal", "mul": True, "home": False, "via": "mistral.rs AFQ/MLX"},
    "fp16": {"bits": 16.0, "device": "gpu", "mul": True, "home": False},
}


def pick_backend(watts_budget=45.0, has_gguf=False, metal=False):
    if watts_budget <= 45.0 and not has_gguf:
        return "bitnet_158"
    if metal and has_gguf:
        return "afq4"
    if has_gguf:
        return "q4k"
    return "bitnet_158"


def absmean(xs):
    return (sum(abs(x) for x in xs) / len(xs)) if xs else 0.0


def quant_158(xs):
    s = absmean(xs)
    if s <= 1e-12:
        return [0] * len(xs), 0.0
    out = []
    for x in xs:
        v = x / s
        out.append(1 if v > 0.5 else (-1 if v < -0.5 else 0))
    return out, s


def pack_2bit(trits):
    acc = 0
    n = 0
    out = bytearray()
    for t in trits:
        bits = 1 if t < 0 else (2 if t > 0 else 0)
        acc |= bits << (n * 2)
        n += 1
        if n == 4:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc)
    return bytes(out)


def matvec(w, rows, cols, x, scale):
    y = [0.0] * rows
    for r in range(rows):
        a = 0.0
        base = r * cols
        for c in range(cols):
            t = w[base + c]
            if t:
                a += x[c] if t > 0 else -x[c]
        y[r] = a * scale
    return y


class Envelope:
    def __init__(self, port="JuniorBitNetFieldCore", backend="bitnet_158"):
        self.port = port
        self.backend = backend if backend in BACKENDS else "bitnet_158"
        self.tokens = []
        self.trit = 0
        self.margin = 0.0

    def as_dict(self):
        spec = BACKENDS[self.backend]
        return {
            "port": self.port,
            "backend": self.backend,
            "bits": spec["bits"],
            "mul_free": not spec["mul"],
            "home_ok": spec["home"],
            "trit": self.trit,
            "margin": round(self.margin, 5),
            "tokens": list(self.tokens),
            "note": "intent only; no POS write; no knockback write",
        }


class Sandbox:
    ACTIONS = ("hold", "move", "flag", "viz", "stock", "halt")

    def __init__(self, backend="bitnet_158"):
        self.env = Envelope(backend=backend)
        rows, cols = 6, 16
        w = [0] * (rows * cols)
        for row, cols_i in ((0, (0, 1)), (1, (2, 3)), (2, (4,)), (3, (8, 9)), (4, (5, 6)), (5, (7,))):
            for c in cols_i:
                w[row * cols + c] = 1
        self.w = w
        self.rows, self.cols = rows, cols
        self.scale = 1.0
        self.last = "halt"

    def eval(self, feat, hyst=0.08):
        x = (list(feat) + [0.0] * self.cols)[: self.cols]
        logits = matvec(self.w, self.rows, self.cols, x, self.scale)
        ranked = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
        top, second = ranked[0], ranked[1]
        margin = logits[top] - logits[second]
        action = self.ACTIONS[top]
        if margin < hyst and self.last != "halt":
            action = self.last
        self.last = action
        trit = 1 if logits[top] > 0 else (-1 if logits[top] < 0 else 0)
        self.env.trit = trit
        self.env.margin = margin
        self.env.tokens = ["intent", action, trit, self.env.backend]
        return {
            "action": action,
            "logits": [round(v, 5) for v in logits],
            "margin": round(margin, 5),
            "packed_bytes": len(pack_2bit(self.w)),
            "envelope": self.env.as_dict(),
        }


def self_test():
    sb = Sandbox()
    a = sb.eval([0.9, 0.4] + [0.0] * 14)
    b = sb.eval([0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.7] + [0.0] * 9)
    c = sb.eval([0.0] * 8 + [0.9, 0.6] + [0.0] * 6)
    assert a["action"] == "hold", a
    assert b["action"] == "stock", b
    assert c["action"] == "viz", c
    assert pick_backend(45.0) == "bitnet_158"
    assert pick_backend(90.0, has_gguf=True) == "q4k"
    return {"ok": True, "a": a["action"], "b": b["action"], "c": c["action"], "bytes": a["packed_bytes"]}
