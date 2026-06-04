# path: src/bitnet/quantization_utils.py

"""
Efficient Quantization Utilities for BitNet 3.0 and custom JuniorLLM.

These utilities support scaling by enabling:
- Low-memory inference on desktop/edge (M-series, CUDA, CPU)
- Autonomous agents that run 24/7 without cloud dependency
- Profile-specific adaptation without full model retraining
- Physics-informed spatial computing on resource-constrained hardware

Utility cases that warrant scaling:
- On-device multi-agent systems (JuniorAGI)
- Real-time business logic (POS, finance, maintenance in JuniorClimbs)
- Long-running sovereign nodes with persistent state + adapters
- Efficient multimodal conversion via TopologySurgeon + protected layers
"""

from typing import Any, Dict, Optional
import mlx.core as mx


def estimate_memory_savings(base_params: int, adapter_rank: int = 8, num_adapters: int = 3) -> Dict[str, float]:
    """Estimate memory savings from using LowRankAdapter on ternary base vs full fine-tuning."""
    base_memory_fp16 = base_params * 2  # bytes
    adapter_memory = num_adapters * (base_params * adapter_rank * 2 * 2)  # rough
    ternary_base = base_params * 0.1875  # ~1.58-bit effective

    full_finetune = base_memory_fp16 * 1.1
    adapter_only = ternary_base + adapter_memory

    savings = (full_finetune - adapter_only) / full_finetune
    return {
        "base_ternary_mb": round(ternary_base / 1e6, 2),
        "adapter_only_mb": round(adapter_only / 1e6, 2),
        "full_finetune_mb": round(full_finetune / 1e6, 2),
        "savings_percent": round(savings * 100, 1)
    }


def apply_mixed_precision(ternary_tensor: mx.array, profile: str = "general") -> mx.array:
    """Simple mixed-precision hook. Future: profile-specific scaling."""
    if profile == "spatial":
        # Higher precision for spatial reasoning if needed
        return ternary_tensor.astype(mx.float16)
    return ternary_tensor


def get_quantization_stats(tensor: mx.array) -> Dict[str, float]:
    """Basic stats for monitoring quantization health."""
    return {
        "mean_abs": float(mx.mean(mx.abs(tensor))),
        "sparsity": float(mx.mean(tensor == 0)),
        "max_val": float(mx.max(mx.abs(tensor))),
    }
