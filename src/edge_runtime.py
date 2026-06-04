# path: src/juniorhome/edge_runtime.py
#!/usr/bin/env python3
"""
EdgeRuntime (Enhanced with TriState + Kernel Awareness)

Now considers Tri-State execution mode and kernel injection
when making efficiency decisions.
"""

import logging
from typing import Any, Callable, Dict, Optional

try:
    from .junioros.kernel_bridge import JuniorOSKernelBridge
    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class EdgeRuntime:
    def __init__(self, max_memory_mb: int = 2048, prefer_quantized: bool = True):
        from .edge_compute_manager import EdgeComputeManager
        from .memory_manager import MemoryManager

        self.edge_manager = EdgeComputeManager(prefer_quantized=prefer_quantized)
        self.memory_manager = MemoryManager(max_memory_mb=max_memory_mb)
        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL else None

        logging.info("EdgeRuntime initialized with TriState + kernel awareness")

    def should_use_quantized(self, task_complexity: str = "medium", available_memory_mb: Optional[int] = None) -> bool:
        return self.edge_manager.should_use_quantized(task_complexity, available_memory_mb)

    def execute_efficiently(
        self,
        func: Callable,
        task_name: str = "operation",
        estimated_memory_mb: int = 50,
        task_complexity: str = "medium",
        inject_to_kernel: bool = False,
    ) -> Any:
        if not self.memory_manager.can_allocate(estimated_memory_mb):
            logging.warning(f"Insufficient memory for {task_name}. Skipping or falling back.")
            return None

        self.memory_manager.allocate(task_name, estimated_memory_mb)

        use_quantized = self.should_use_quantized(task_complexity)

        try:
            result = self.edge_manager.execute_efficiently(
                func,
                use_quantized=use_quantized,
                task_complexity=task_complexity,
            )

            # Optional kernel injection after efficient execution
            if inject_to_kernel and self.kernel_bridge and self.kernel_bridge.is_available():
                try:
                    # Best effort injection
                    self.kernel_bridge.write_ternary_manifold(
                        ternary_tensor=result if hasattr(result, "__array__") else None,
                        metadata={"task": task_name},
                        coherence=0.0,
                    )
                except Exception:
                    pass

            return result
        finally:
            self.memory_manager.release(task_name)

    def get_status(self) -> Dict[str, Any]:
        status = {
            "edge_compute": self.edge_manager.get_stats(),
            "memory": self.memory_manager.get_usage(),
        }
        if self.kernel_bridge:
            status["kernel_available"] = self.kernel_bridge.is_available()
        return status
