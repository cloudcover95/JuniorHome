# path: src/juniorhome/edge_compute_manager.py
#!/usr/bin/env python3
"""
Edge Compute Manager (Enhanced)

More sophisticated edge efficiency management with dynamic backend selection,
resource awareness, and integration with quantized models.
"""

import logging
from typing import Any, Callable, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EdgeComputeManager:
    def __init__(self, prefer_quantized: bool = True, max_batch_size: int = 8):
        self.prefer_quantized = prefer_quantized
        self.max_batch_size = max_batch_size
        self.stats = {
            "quantized_calls": 0,
            "full_precision_calls": 0,
            "batched_operations": 0,
            "total_operations": 0,
        }
        logging.info("EdgeComputeManager (enhanced) initialized")

    def should_use_quantized(self, task_complexity: str = "medium", available_memory_mb: Optional[int] = None) -> bool:
        if not self.prefer_quantized:
            return False

        if available_memory_mb is not None and available_memory_mb < 512:
            return True  # Force quantized on very low memory

        if task_complexity == "low":
            return True
        if task_complexity == "high":
            return False
        return True  # Default to quantized for edge

    def get_optimal_batch_size(self, pending_items: int, available_memory_mb: Optional[int] = None) -> int:
        batch = min(pending_items, self.max_batch_size)
        if available_memory_mb is not None and available_memory_mb < 1024:
            batch = min(batch, 4)
        return batch

    def execute_efficiently(
        self,
        func: Callable,
        use_quantized: Optional[bool] = None,
        batch: bool = False,
        task_complexity: str = "medium",
    ) -> Any:
        if use_quantized is None:
            use_quantized = self.should_use_quantized(task_complexity)

        self.stats["total_operations"] += 1

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
