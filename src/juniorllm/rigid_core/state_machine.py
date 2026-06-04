# path: src/juniorhome/juniorllm/rigid_core/state_machine.py
#!/usr/bin/env python3
"""
JuniorLLM Rigid Core State Machine (Expanded - Hierarchical Spatial + Persistence)

Major updates this iteration:
- Stronger hierarchical spatial state support
- Deeper persistence (state + active timers + spatial manifold metadata)
- Better integration hooks for generalized Ternary Spatial Manifolds
- Foundation for evolution rules and long-term spatial tracking

Designed to be domain-agnostic (not market-specific).
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

try:
    from ...junioros.kernel_bridge import JuniorOSKernelBridge
    HAS_KERNEL_BRIDGE = True
except ImportError:
    HAS_KERNEL_BRIDGE = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class State(Enum):
    IDLE = auto()
    ACTIVE_INFERENCE = auto()
    MAINTENANCE = auto()
    EVOLUTION = auto()
    SPATIAL_MONITORING = auto()     # General spatial manifold monitoring
    SPATIAL_EVOLUTION = auto()      # Controlled evolution of spatial state


class SpatialSubState(Enum):
    """Hierarchical sub-states under SPATIAL_EVOLUTION / SPATIAL_MONITORING."""
    INITIALIZE = auto()
    TRACK = auto()
    DETECT_DRIFT = auto()
    EVOLVE = auto()
    FUSE = auto()           # Multi-node / UDP mesh fusion
    COMPLETE = auto()


@dataclass
class Timer:
    name: str
    interval: float
    last_triggered: float = field(default_factory=time.time)
    persistent: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_fire(self) -> bool:
        return (time.time() - self.last_triggered) >= self.interval

    def fire(self):
        self.last_triggered = time.time()


class JuniorLLMStateMachine:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.current_state: State = State.IDLE
        self.current_spatial_sub_state: Optional[SpatialSubState] = None
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL_BRIDGE else None

        # Default persistent timers
        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("spatial_health_check", interval_seconds=600, metadata={"type": "spatial"})

        logging.info(f"JuniorLLMStateMachine initialized for node {node_id}")

    def transition_to(self, new_state: State, reason: str = ""):
        if new_state != self.current_state:
            self.history.append({
                "from": self.current_state.name,
                "to": new_state.name,
                "timestamp": time.time(),
                "reason": reason
            })
            logging.info(f"State transition: {self.current_state.name} -> {new_state.name} ({reason})")
            self.current_state = new_state
            self.current_spatial_sub_state = None
            self._persist_state()

    def transition_spatial_sub_state(self, new_sub_state: SpatialSubState, reason: str = ""):
        if self.current_state not in (State.SPATIAL_MONITORING, State.SPATIAL_EVOLUTION):
            self.transition_to(State.SPATIAL_MONITORING, reason="spatial_sub_state_entry")

        if new_sub_state != self.current_spatial_sub_state:
            self.history.append({
                "from_spatial_sub": self.current_spatial_sub_state.name if self.current_spatial_sub_state else None,
                "to_spatial_sub": new_sub_state.name,
                "timestamp": time.time(),
                "reason": reason
            })
            logging.info(f"Spatial sub-state transition: {self.current_spatial_sub_state} -> {new_sub_state} ({reason})")
            self.current_spatial_sub_state = new_sub_state
            self._persist_state()

    def get_current_state(self) -> State:
        return self.current_state

    def get_current_spatial_sub_state(self) -> Optional[SpatialSubState]:
        return self.current_spatial_sub_state

    # --- Timer Management ---
    def add_timer(self, name: str, interval_seconds: float, persistent: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.timers[name] = Timer(
            name=name,
            interval=interval_seconds,
            persistent=persistent,
            metadata=metadata or {}
        )

    def check_timers(self):
        for timer in list(self.timers.values()):
            if timer.should_fire():
                timer.fire()
                self._handle_timer_event(timer)

    def _handle_timer_event(self, timer: Timer):
        if timer.metadata.get("type") == "spatial":
            self.emit_event("spatial_health_check")
        else:
            self.emit_event(timer.name)

    # --- Event System ---
    def emit_event(self, event_name: str, payload: Any = None):
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                handler(payload)

    def on(self, event_name: str, handler: Callable):
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    # --- Persistence (Deeper) ---
    def _persist_state(self):
        if self.kernel_bridge and self.kernel_bridge.is_available():
            try:
                self.kernel_bridge.write_ternary_manifold(
                    ternary_tensor=None,
                    metadata={
                        "node_id": self.node_id,
                        "current_state": self.current_state.name,
                        "current_spatial_sub_state": self.current_spatial_sub_state.name if self.current_spatial_sub_state else None,
                        "history_length": len(self.history),
                        "active_timers": list(self.timers.keys()),
                        "timestamp": time.time(),
                    },
                    coherence=0.0,
                )
            except Exception as e:
                logging.warning(f"Failed to persist state: {e}")

    # --- Public API ---
    def process_command(self, command: str, payload: Any = None):
        if command == "start_inference":
            self.transition_to(State.ACTIVE_INFERENCE, reason="external_command")
        elif command == "enter_maintenance":
            self.transition_to(State.MAINTENANCE, reason="external_command")
        elif command == "trigger_evolution":
            self.transition_to(State.EVOLUTION, reason="external_command")
        elif command == "monitor_spatial":
            self.transition_to(State.SPATIAL_MONITORING, reason="external_command")
        elif command == "evolve_spatial_manifold":
            self.transition_to(State.SPATIAL_EVOLUTION, reason="external_command")
            self.transition_spatial_sub_state(SpatialSubState.INITIALIZE)
        else:
            self.emit_event(command, payload)

    def handle_spatial_task(self, task_type: str):
        """Handler for general spatial manifold tasks."""
        if task_type == "monitor_spatial_manifold":
            self.transition_to(State.SPATIAL_MONITORING, reason="task")
            return {"status": "spatial_monitoring_started"}

        elif task_type == "evolve_spatial_manifold":
            self.transition_to(State.SPATIAL_EVOLUTION, reason="task")
            self.transition_spatial_sub_state(SpatialSubState.INITIALIZE)
            return {"status": "evolution_started"}

        elif task_type == "fuse_spatial_states":
            if self.current_state == State.SPATIAL_EVOLUTION:
                self.transition_spatial_sub_state(SpatialSubState.FUSE)
            return {"status": "fusion_initiated"}

        elif task_type == "detect_spatial_drift":
            if self.current_state == State.SPATIAL_EVOLUTION:
                self.transition_spatial_sub_state(SpatialSubState.DETECT_DRIFT)
            return {"status": "drift_detection_started"}

        return {"status": "unknown_spatial_task"}


if __name__ == "__main__":
    sm = JuniorLLMStateMachine(node_id="llm_core_01")
    sm.on("spatial_health_check", lambda p: print("[EVENT] Checking spatial manifold health..."))

    print("JuniorLLMStateMachine with hierarchical spatial support running...")
    while True:
        sm.check_timers()
        time.sleep(5)
