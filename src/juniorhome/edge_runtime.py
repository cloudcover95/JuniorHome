# path: src/juniorhome/edge_runtime.py
#!/usr/bin/env python3
"""
Edge Runtime

Unified coordinator for efficient, sovereign edge execution.
Brings together EdgeComputeManager, MemoryManager, TinyMLBridge,
and quantized models into one high-level runtime.

This is the central piece for running the full stack efficiently on-device.
"""

import logging
from typing import Any, Callable, Dict, Optional

from .edge_compute_manager import EdgeComputeManager
from .memory_manager import MemoryManager
from .tinyml_bridge import TinyMLBridge
from .quantized_model_manager import QuantizedModelManager

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EdgeRuntime:
    """
    High-level runtime for efficient sovereign edge compute.
    """

    def __init__(self, max_memory_mb: int = 2048):
        self.edge_manager = EdgeComputeManager(prefer_quantized=True)
        self.memory_manager = MemoryManager(max_memory_mb=max_memory_mb)
        self.tinyml = TinyMLBridge()
        self.quantized_models = QuantizedModelManager()

        logging.info("EdgeRuntime initialized (unified edge execution coordinator)")

    def execute_efficiently(
        self,
        func: Callable,
        task_name: str = "operation",
        estimated_memory_mb: int = 50,
        task_complexity: str = "medium",
    ) -> Any:
        if not self.memory_manager.can_allocate(estimated_memory_mb):
            logging.warning(f"Insufficient memory for {task_name}. Skipping or falling back.")
            return None

        self.memory_manager.allocate(task_name, estimated_memory_mb)

        use_quantized = self.edge_manager.should_use_quantized(task_complexity)

        try:
            result = self.edge_manager.execute_efficiently(
                func,
                use_quantized=use_quantized,
                task_complexity=task_complexity,
            )
            return result
        finally:
            self.memory_manager.release(task_name)

    def load_quantized_model(self, model_name: str = "google/gemma-2-2b") -> bool:
        return self.quantized_models.load_gemma_ternary(model_name)

    def get_status(self) -> Dict[str, Any]:
        return {
            "edge_compute": self.edge_manager.get_stats(),
            "memory": self.memory_manager.get_usage(),
            "quantized_models": self.quantized_models.list_loaded_models(),
        }
