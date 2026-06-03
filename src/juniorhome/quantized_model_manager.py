# path: src/juniorhome/quantized_model_manager.py
#!/usr/bin/env python3
"""
Quantized Model Manager

High-level manager for loading, caching, and using ternary-quantized
models (via BitNet-mlx) inside agents and pipelines.

This makes it easy for AutonomousAgent, KnowledgeService, etc.
to leverage 1.58-bit models like Gemma.
"""

import logging
from typing import Any, Dict, Optional

from .quantized_model_integration import QuantizedModelIntegration

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class QuantizedModelManager:
    """
    High-level manager for ternary-quantized models in the agent ecosystem.
    """

    def __init__(self):
        self.integration = QuantizedModelIntegration()
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        logging.info("QuantizedModelManager initialized")

    def load_gemma_ternary(self, model_name: str = "google/gemma-2-2b") -> bool:
        success = self.integration.load_gemma_ternary(model_name)
        if success:
            self.loaded_models[model_name] = self.integration.get_model(model_name)
        return success

    def get_model(self, name: str) -> Optional[Any]:
        return self.integration.get_model(name)

    def get_report(self, name: str) -> Optional[Dict[str, Any]]:
        return self.integration.get_quantization_report(name)

    def list_loaded_models(self) -> list:
        return list(self.loaded_models.keys())

    def is_model_loaded(self, name: str) -> bool:
        return name in self.loaded_models
