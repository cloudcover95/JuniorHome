# path: src/bitnet/converter/surgeon.py
# Integrated from advanced ecosystem capabilities
# (TopologySurgeon for multi-modal bypass)

import json
import shutil
import glob
import logging
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
from mlx_lm.utils import load, save_weights
from huggingface_hub import snapshot_download

logger = logging.getLogger("TopologySurgeon")

class TopologySurgeon:
    @staticmethod
    def transmute(module: nn.Module) -> nn.Module:
        protected_gates = ["lm_head", "embed_tokens", "embed", "gate", "vision_proj", "multi_modal", "image_encoder", "conv", "wte", "wpe", "norm", "ln_", "audio_encoder", "feature_extractor", "mel"]
        def _replace(mod: nn.Module, prefix: str = ""):
            for name, child in list(mod.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear) and not any(k in path.lower() for k in protected_gates):
                    in_dim = child.weight.shape[1]
                    out_dim = child.weight.shape[0]
                    has_bias = child.bias is not None
                    from ..core.dynamic_bitlinear import DynamicBitLinear
                    new_layer = DynamicBitLinear(in_features=in_dim, out_features=out_dim, bias=has_bias)
                    new_layer.weight = child.weight
                    if has_bias:
                        new_layer.bias = child.bias
                    setattr(mod, name, new_layer)
                else:
                    _replace(child, path)
        _replace(module)
        return module

    @staticmethod
    def build_manifold(repo_id: str, output_dir: str) -> None:
        try:
            logger.info(f"Ingesting FP16 topology from {repo_id}...")
            model, tokenizer = load(repo_id)
            quantized_model = TopologySurgeon.transmute(model)
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            from mlx_lm.utils import save_weights
            save_weights(out_path, dict(quantized_model.parameters()))
            logger.info(f"Matrix secured at {output_dir}")
        except Exception as e:
            logger.error(f"Surgical failure: {e}")
            raise
