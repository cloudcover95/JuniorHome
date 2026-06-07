# path: src/juniorllm/training/real_data_runner.py

"""
RealDataRunner

Added efficiency profiling hooks.
"""

from typing import Any, Callable, Dict, List, Optional

import time


class RealDataRunner:
    def __init__(self, plasticity_engine, memsys_store, graph_memory: Optional[Any] = None, theoretical_math_fn: Optional[Callable] = None):
        self.plasticity = plasticity_engine
        self.memsys = memsys_store
        self.graph_memory = graph_memory
        self.theoretical_math_fn = theoretical_math_fn
        self.step_count = 0
        self.spiking_mode = False
        self._start_time = time.time()

    def set_spiking_mode(self, enabled: bool = True):
        self.spiking_mode = enabled
        if hasattr(self.plasticity, "set_spiking_mode"):
            self.plasticity.set_spiking_mode(enabled)

    def process_vision_pattern(self, pattern: Dict[str, Any], outcome: float = 1.0) -> Dict[str, Any]:
        if self.theoretical_math_fn:
            try:
                result = self.theoretical_math_fn(pattern, outcome)
                if isinstance(result, dict):
                    pattern.update(result)
            except Exception as e:
                print(f"[RealDataRunner] Theoretical math error: {e}")

        profile = "vision_" + (pattern.get("detected_tags", ["general"])[0] if pattern.get("detected_tags") else "general")

        self.plasticity.update_eligibility_trace(profile, strength=1.0)
        state = {"active_profile": profile, "performance": {}}
        self.plasticity.apply(
            performance=state["performance"],
            lifecycle={},
            profile=profile,
            outcome=outcome
        )

        if self.memsys:
            try:
                self.memsys.store_vision_pattern(pattern)
            except Exception as e:
                print(f"[RealDataRunner] MemSys error: {e}")

        if self.graph_memory:
            try:
                self.graph_memory.store_pattern({
                    "type": "vision_pattern",
                    "data": pattern,
                    "outcome": outcome
                })
            except Exception as e:
                print(f"[RealDataRunner] GraphMemory error: {e}")

        self.step_count += 1
        return {
            "profile": profile,
            "connection_strength": self.plasticity.get_connection_strength(profile),
            "neuromodulation": self.plasticity.get_neuromodulation(),
            "step": self.step_count,
            "spiking_mode": self.spiking_mode,
            "elapsed_time": time.time() - self._start_time
        }

    def process_call_event(self, event: Dict[str, Any], outcome: float = 1.0) -> Dict[str, Any]:
        if self.memsys:
            try:
                self.memsys.store_call_event(event)
            except Exception as e:
                print(f"[RealDataRunner] MemSys error: {e}")

        if self.graph_memory:
            try:
                self.graph_memory.store_pattern({
                    "type": "call_event",
                    "data": event,
                    "outcome": outcome
                })
            except Exception as e:
                print(f"[RealDataRunner] GraphMemory error: {e}")

        self.step_count += 1
        return {
            "step": self.step_count,
            "event_type": event.get("type"),
            "elapsed_time": time.time() - self._start_time
        }

    def trigger_sleep_consolidation(self):
        if hasattr(self.plasticity, "sleep_consolidation"):
            self.plasticity.sleep_consolidation()

    def get_efficiency_report(self) -> Dict[str, Any]:
        return {
            "steps_processed": self.step_count,
            "neuromodulation_level": self.plasticity.get_neuromodulation() if hasattr(self.plasticity, "get_neuromodulation") else None,
            "spiking_mode": self.spiking_mode,
            "total_runtime": time.time() - self._start_time,
            "avg_time_per_step": (time.time() - self._start_time) / max(self.step_count, 1)
        }
