# path: src/bitnet/backends.py

from enum import Enum
from typing import Any, Optional

try:
    from .hardware.hardware_capability import HardwareCapability
    HAS_HARDWARE_CAPABILITY = True
except ImportError:
    HAS_HARDWARE_CAPABILITY = False
    HardwareCapability = None


class Backend(Enum):
    MLX = "mlx"
    CUDA = "cuda"


class BackendRouter:
    """
    Hardware-agnostic backend router for BitNet/ternary workloads.
    Supports routed AGI with capacity awareness.
    """

    def __init__(self):
        self._current = Backend.MLX
        self._matmul_impl = None
        self.hardware_cap = HardwareCapability() if HAS_HARDWARE_CAPABILITY else None
        self.set_backend(Backend.MLX)

    def set_backend(self, backend: Backend):
        self._current = backend
        if backend == Backend.MLX:
            from .kernels.ternary_kernels import ternary_matmul as impl
        elif backend == Backend.CUDA:
            from .kernels.cuda_kernels import cuda_ternary_matmul as impl
        else:
            raise ValueError("Unsupported backend")
        self._matmul_impl = impl

    def ternary_matmul(self, input: Any, ternary_weight: Any, scale: Any):
        return self._matmul_impl(input, ternary_weight, scale)

    def suggest_routing(self, estimated_agent_mb: float = 450) -> dict:
        if self.hardware_cap:
            cap = self.hardware_cap.estimate_max_lightweight_agents(
                self._current.name.lower(), estimated_agent_mb
            )
            return {
                "backend": self._current.name,
                "estimated_max_agents": cap
            }
        return {"backend": self._current.name}


# Global router instance
router = BackendRouter()
