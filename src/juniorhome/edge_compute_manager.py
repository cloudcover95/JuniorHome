# path: src/juniorhome/edge_compute_manager.py
#!/usr/bin/env python3
"""
Edge Compute Manager

Manages efficient execution on edge devices.
Helps the system make smart decisions about when to use
quantized models, batch operations, and minimize resource usage.

Part of building a deeply efficient sovereign edge architecture.
"""

import logging
from typing import Any, Callable, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EdgeComputeManager:
    """
    Helps orchestrate compute-efficient behavior on edge hardware.
    """

    def __init__(self, prefer_quantized: bool = True, max_batch_size: int = 8):
        self.prefer_quantized = prefer_quantized
        self.max_batch_size = max_batch_size
        self.stats = {
            "quantized_calls": 0,
            "full_precision_calls": 0,
            "batched_operations": 0,
        }
        logging.info("EdgeComputeManager initialized")

    def should_use_quantized(self, task_complexity: str = "medium") -> bool:
        if not self.prefer_quantized:
            return False

        # Simple heuristic: use quantized for most tasks on edge
        if task_complexity in ["low", "medium"]:
            return True
        return False

    def get_optimal_batch_size(self, pending_items: int) -> int:
        return min(pending_items, self.max_batch_size)

    def execute_efficiently(
        self,
        func: Callable,
        use_quantized: Optional[bool] = None,
        batch: bool = False,
    ) -> Any:
        if use_quantized is None:
            use_quantized = self.prefer_quantized

        if use_quantized:
            self.stats["quantized_calls"] += 1
        else:
            self.stats["full_precision_calls"] += 1

        if batch:
            self.stats["batched_operations"] += 1

        return func()

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

    def reset_stats(self):
        self.stats = {k: 0 for k in self.stats}
