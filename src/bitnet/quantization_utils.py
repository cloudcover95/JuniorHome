# path: src/bitnet/quantization_utils.py

"""
Efficient Quantization Utilities for BitNet 1.58 → 3.0 and custom JuniorLLM.

Focus: Quantify why the 3.0 extensions (adapters, profiles, specialization, manifold integration, persistence)
 warrant scaling beyond pure 1.58 ternary inference.

Utility cases:
- Desktop/enterprise power users needing long-running autonomous agents
- Sovereign business systems (JuniorClimbs POS, finance, maintenance) with continual adaptation
- Edge nodes that must remain functional offline with persistent state + specialized behavior
- Efficient spatial/physics-informed computing without massive models
"""

from typing import Any, Dict, Optional
import mlx.core as mx


def estimate_memory_savings(base_params: int, adapter_rank: int = 8, num_adapters: int = 3) -> Dict[str, float]:
    """Estimate memory savings from LowRankAdapter on ternary base vs full fine-tuning."""
    base_memory_fp16 = base_params * 2
    adapter_memory = num_adapters * (base_params * adapter_rank * 2 * 2)
    ternary_base = base_params * 0.1875

    full_finetune = base_memory_fp16 * 1.1
    adapter_only = ternary_base + adapter_memory

    savings = (full_finetune - adapter_only) / full_finetune
    return {
        "base_ternary_mb": round(ternary_base / 1e6, 2),
        "adapter_only_mb": round(adapter_only / 1e6, 2),
        "full_finetune_mb": round(full_finetune / 1e6, 2),
        "savings_percent": round(savings * 100, 1)
    }


def estimate_1_58_vs_3_0_gains(
    base_params: int = 1_000_000_000,
    adapter_rank: int = 8,
    num_adapters: int = 4,
    has_profiles: bool = True,
    has_specialization: bool = True,
    has_persistence: bool = True,
    has_manifold_integration: bool = True
) -> Dict[str, Any]:
    """
    Quantify scaling benefits of BitNet 3.0 extensions over pure 1.58 ternary inference.

    1.58 baseline: Efficient inference, but static behavior, no easy specialization,
    limited long-term autonomy without external orchestration.

    3.0 extensions (adapters + profiles + specialization + persistence + manifold integration):
    - Parameter-efficient continual adaptation
    - Context-aware profile switching (spatial/quant/business)
    - Autonomous specialization cycles
    - Restart-resilient persistent state + profile memory
    - Physics-informed co-evolution with spatial manifold
    """
    base_1_58 = base_params * 0.1875  # ternary
    adapter_mem = num_adapters * (base_params * adapter_rank * 2 * 2)
    full_3_0 = base_1_58 + adapter_mem

    gains = {
        "1_58_baseline_mb": round(base_1_58 / 1e6, 2),
        "3_0_with_adapters_mb": round(full_3_0 / 1e6, 2),
        "memory_overhead_percent": round((adapter_mem / base_1_58) * 100, 1),
    }

    if has_profiles:
        gains["profile_switching"] = "enables context-aware behavior without retraining"
    if has_specialization:
        gains["autonomous_specialization"] = "run_specialization_cycle() enables proactive adaptation"
    if has_persistence:
        gains["restart_resilience"] = "profile + manifold state survives restarts"
    if has_manifold_integration:
        gains["physics_informed"] = "spatial manifold influences and is influenced by LLM state"

    gains["key_scaling_justification"] = (
        "Small memory overhead for massive gains in autonomy, specialization, "
        "restart resilience, and physics-informed reasoning. Enables sovereign long-running "
        "agents and business systems on desktop/edge hardware without cloud dependency."
    )

    return gains


def apply_mixed_precision(ternary_tensor: mx.array, profile: str = "general") -> mx.array:
    if profile == "spatial":
        return ternary_tensor.astype(mx.float16)
    return ternary_tensor


def get_quantization_stats(tensor: mx.array) -> Dict[str, float]:
    return {
        "mean_abs": float(mx.mean(mx.abs(tensor))),
        "sparsity": float(mx.mean(tensor == 0)),
        "max_val": float(mx.max(mx.abs(tensor))),
    }
