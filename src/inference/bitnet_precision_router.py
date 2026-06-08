# path: src/inference/bitnet_precision_router.py

"""
BitNetPrecisionRouter

Lean, efficient mode selector for BitNet inference.

- Default: 1.58-bit ternary (maximum efficiency on edge / M4 / future NVIDIA Spark)
- Escalate: Higher precision (e.g. 3.0 or FP16) only when coherence or task criticality requires it.

This is called by the application layer (JuniorQuant / JuniorStock) — not dictated by it.
BitNet-mlx remains the engine that implements the actual quantization.

Design goals: Near-zero overhead, black-box friendly, plasticity-aware.
"""

from typing import Any, Dict, Optional


class BitNetPrecisionRouter:
    """
    Lightweight router that decides 1.58-bit vs higher precision.

    Used by JuniorQuant-style trading agents and other high-level components.
    """

    def __init__(self, default_mode: str = "1.58", coherence_threshold: float = 0.65):
        self.default_mode = default_mode          # "1.58" or "higher"
        self.coherence_threshold = coherence_threshold
        self._last_mode = default_mode

    def select_mode(
        self,
        coherence: Optional[float] = None,
        task_criticality: float = 0.5,
        plasticity_signal: Optional[Dict[str, Any]] = None,
        force_mode: Optional[str] = None
    ) -> str:
        """
        Lean decision logic.

        Returns "1.58" (default for efficiency) or "higher".

        Rules (kept minimal for speed):
        - If force_mode is given, respect it.
        - If coherence is low or task is highly critical, escalate.
        - Otherwise stay in efficient 1.58-bit mode.
        """
        if force_mode in ("1.58", "higher"):
            self._last_mode = force_mode
            return force_mode

        # Default to efficient mode
        mode = self.default_mode

        if coherence is not None and coherence < self.coherence_threshold:
            mode = "higher"

        if task_criticality > 0.85:
            mode = "higher"

        # Optional: use plasticity signal strength as additional signal
        if plasticity_signal:
            conn_strength = plasticity_signal.get("connection_strength", 0.0)
            if conn_strength < 0.3:   # low confidence in learned behavior
                mode = "higher"

        self._last_mode = mode
        return mode

    def get_last_mode(self) -> str:
        return self._last_mode

    def reset(self):
        self._last_mode = self.default_mode


# Convenience function for quick use in JuniorQuant-style code
def select_bitnet_mode(
    coherence: Optional[float] = None,
    task_criticality: float = 0.5,
    plasticity_signal: Optional[Dict[str, Any]] = None,
    force_mode: Optional[str] = None
) -> str:
    router = BitNetPrecisionRouter()
    return router.select_mode(
        coherence=coherence,
        task_criticality=task_criticality,
        plasticity_signal=plasticity_signal,
        force_mode=force_mode
    )
