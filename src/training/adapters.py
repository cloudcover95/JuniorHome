# path: src/training/adapters.py
import math
import mlx.core as mx
import mlx.nn as nn
from ..core.dynamic_bitlinear import DynamicBitLinear

class LowRankAdapter(nn.Module):
    """Universal low-rank proxy for LoRA/QLoRA on ternary manifolds."""
    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.05, adapter_id: str = None):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = alpha / r
        self.adapter_id = adapter_id or f"lora_{id(self)}"

        in_d, out_d = base_layer.weight.shape[1], base_layer.weight.shape[0]
        self.lora_a = mx.random.uniform(low=-1/math.sqrt(in_d), high=1/math.sqrt(in_d), shape=(in_d, r), dtype=mx.float16)
        self.lora_b = mx.zeros((r, out_d), dtype=mx.float16)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base_layer(x)
        lora_out = mx.matmul(self.dropout(x), self.lora_a)
        lora_out = mx.matmul(lora_out, self.lora_b) * self.scaling
        return base_out + lora_out

    def state_dict(self, *args, **kwargs):
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if "lora_" in k}

    def load_state_dict(self, state_dict, strict=False):
        adapter_state = {k: v for k, v in state_dict.items() if "lora_" in k}
        return super().load_state_dict(adapter_state, strict=False)

def inject_adapters(model: nn.Module, r: int = 8, alpha: float = 16.0) -> nn.Module:
    """Surgically routes LowRank wrappers around all DynamicBitLinear manifolds."""
    def _replace(mod: nn.Module, prefix: str = ""):
        for name, child in list(mod.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, DynamicBitLinear):
                setattr(mod, name, LowRankAdapter(child, r=r, alpha=alpha))
            else:
                _replace(child, path)
    _replace(model)
    return model
