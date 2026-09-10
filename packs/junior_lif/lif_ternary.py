"""Ternary LIF — JuniorHome pack. Stdlib only."""
from __future__ import annotations
from dataclasses import dataclass, field

def fire_ternary(u, theta):
    if theta <= 0: return 0
    if u >= theta: return 1
    if u <= -theta: return -1
    return 0

def reset_u(u, spike, theta, mode):
    if spike == 0: return u
    if mode == "zero": return 0.0
    return u - float(spike) * theta

@dataclass
class LifLayer:
    rows: int
    cols: int
    weights: list
    scale: float = 1.0
    leak: float = 0.75
    theta: float = 1.0
    reset: str = "subtract"
    u: list = field(default_factory=list)
    def __post_init__(self):
        need = self.rows * self.cols
        w = list(self.weights) + [0] * max(0, need - len(self.weights))
        self.weights = w[:need]
        if len(self.u) != self.rows: self.u = [0.0] * self.rows
    def step(self, x):
        out = [0] * self.rows
        for r in range(self.rows):
            acc = 0.0
            base = r * self.cols
            n = min(self.cols, len(x))
            for j in range(n):
                w = self.weights[base + j]
                if w: acc += x[j] if w > 0 else -x[j]
            self.u[r] = self.leak * self.u[r] + acc * self.scale
            s = fire_ternary(self.u[r], self.theta)
            self.u[r] = reset_u(self.u[r], s, self.theta, self.reset)
            out[r] = s
        return out
    def reset_state(self):
        self.u = [0.0] * self.rows

def encode_features(feat, ticks, layer):
    layer.reset_state()
    acc = [0.0] * layer.rows
    xs = max(1, ticks)
    for _ in range(xs):
        for r, v in enumerate(layer.step(feat)):
            acc[r] += float(v)
    return [v / xs for v in acc]

def demo_layer(rows=8, cols=16, seed=3):
    w = []; s = seed
    for _ in range(rows * cols):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        w.append((s % 3) - 1)
    return LifLayer(rows=rows, cols=cols, weights=w, scale=0.53, leak=0.8, theta=1.0)
