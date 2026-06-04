# path: src/bitnet/kernels/metal_kernels.py

import mlx.core as mx


def ternary_matmul_metal(input: mx.array, ternary_weight: mx.array, scale: mx.array) -> mx.array:
    """
    Target location for custom MLX Metal ternary matmul kernel.
    Future: implement real Metal Shading Language kernel for maximum performance
    on Apple Silicon using packed int8 weights + per-channel scale.
    """
    # Current fallback
    w = ternary_weight.astype(input.dtype) * scale
    return mx.matmul(input, w.T)
