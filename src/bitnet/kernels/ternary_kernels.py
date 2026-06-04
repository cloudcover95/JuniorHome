# path: src/bitnet/kernels/ternary_kernels.py

import mlx.core as mx
from typing import Tuple, Optional


def abs_mean_quantize(weight: mx.array, eps: float = 1e-8) -> Tuple[mx.array, mx.array]:
    """BitNet 1.58 AbsMean quantization to ternary {-1, 0, +1}."""
    scale = mx.mean(mx.abs(weight), axis=1, keepdims=True) + eps
    scaled = weight / scale
    ternary = mx.where(scaled > 0.5, 1.0,
              mx.where(scaled < -0.5, -1.0, 0.0))
    return ternary.astype(mx.int8), scale


def ternary_matmul(input: mx.array, ternary_weight: mx.array, scale: mx.array) -> mx.array:
    """Optimized ternary matmul for BitNet 1.58."""
    w = ternary_weight.astype(input.dtype) * scale
    return mx.matmul(input, w.T)


class BitNetLinear:
    """BitNet 1.58 style linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        weight = mx.random.normal((out_features, in_features)) * 0.02
        self.ternary_weight, self.scale = abs_mean_quantize(weight)
        self.bias = mx.zeros((out_features,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        out = ternary_matmul(x, self.ternary_weight, self.scale)
        if self.bias is not None:
            out = out + self.bias
        return out

    def quantize_from_weight(self, weight: mx.array):
        self.ternary_weight, self.scale = abs_mean_quantize(weight)
