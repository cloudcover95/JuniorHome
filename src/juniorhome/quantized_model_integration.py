# path: src/juniorhome/quantized_model_integration.py
#!/usr/bin/env python3
"""
Quantized Model Integration

Allows JuniorHome agents and pipelines to use BitNet-mlx
ternary-quantized models (e.g. Gemma 4B in 1.58-bit).

This bridges the quantization engine into the agent ecosystem.
"""

import logging
from typing import Any, Dict, Optional

try:
    from bitnet_mlx.src.quantization.gemma_ternary_converter import GemmaTernaryConverter
    HAS_GEMMA = True
except ImportError:
    HAS_GEMMA = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class QuantizedModelIntegration:
    """
    Integration point for using ternary-quantized models in agents.
    """

    def __init__(self):
        self.quantized_models: Dict[str, Any] = {}
        logging.info("QuantizedModelIntegration initialized")

    def load_gemma_ternary(self, model_name: str = "google/gemma-2-2b") -> bool:
        if not HAS_GEMMA:
            logging.warning("GemmaTernaryConverter not available")
            return False

        try:
            converter = GemmaTernaryConverter(model_name=model_name)
            report = converter.convert_model()
            self.quantized_models[model_name] = {
                "converter": converter,
                "report": report,
            }
            logging.info(f"Loaded ternary-quantized {model_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to load ternary Gemma: {e}")
            return False

    def get_model(self, name: str) -> Optional[Any]:
        return self.quantized_models.get(name)

    def get_quantization_report(self, name: str) -> Optional[Dict[str, Any]]:
        model_info = self.quantized_models.get(name)
        return model_info.get("report") if model_info else None
