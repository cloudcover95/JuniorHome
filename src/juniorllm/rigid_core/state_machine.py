# path: src/juniorhome/juniorllm/rigid_core/state_machine.py
#!/usr/bin/env python3
"""
JuniorLLM Rigid Core - Hierarchical Event-Driven State Machine (Layer 1)

Initial concrete implementation focusing on:
- Core state machine structure
- Persistent timer management
- Event-driven transitions
- Integration hooks for HighLevelOrchestrator, TriStateExecutionEngine,
  JuniorOSKernelBridge, and SecondBrainPipeline

This forms the rigid foundation for long-term state, timers, and
controlled autonomous evolution without requiring full agent frameworks.
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
    """Root states for the rigid core."""
    IDLE = auto()
    ACTIVE_INFERENCE = auto()
    MAINTENANCE = auto()
    EVOLUTION = auto()


@dataclass
class Timer:
    name: str
    interval: float          # seconds
    last_triggered: float = field(default_factory=time.time)
    persistent: bool = True

    def should_fire(self) -> bool:
        return (time.time() - self.last_triggered) >= self.interval

    def fire(self):
        self.last_triggered = time.time()


class JuniorLLMStateMachine:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.current_state: State = State.IDLE
        self.history: List[State] = []
        self.timers: Dict[str, Timer] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL_BRIDGE else None

        logging.info(f"JuniorLLMStateMachine initialized for node {node_id}")

    # --- State Management ---
    def transition_to(self, new_state: State, reason: str = ""):
        if new_state != self.current_state:
            self.history.append(self.current_state)
            logging.info(f"State transition: {self.current_state.name} -> {new_state.name} ({reason})")
            self.current_state = new_state
            self._persist_state()

    def get_current_state(self) -> State:
        return self.current_state

    # --- Timer Management (Persistent) ---
    def add_timer(self, name: str, interval_seconds: float, persistent: bool = True):
        self.timers[name] = Timer(name=name, interval=interval_seconds, persistent=persistent)
        logging.info(f"Timer '{name}' added with interval {interval_seconds}s")

    def check_timers(self):
        for timer in list(self.timers.values()):
            if timer.should_fire():
                timer.fire()
                self._handle_timer_event(timer.name)

    def _handle_timer_event(self, timer_name: str):
        logging.info(f"Timer fired: {timer_name}")
        # Example: trigger internal events based on timer
        if timer_name == "coherence_check":
            self.emit_event("coherence_check")
        elif timer_name == "maintenance_tick":
            self.emit_event("maintenance_tick")

    # --- Event System ---
    def emit_event(self, event_name: str, payload: Any = None):
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                handler(payload)

    def on(self, event_name: str, handler: Callable):
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    # --- Persistence ---
    def _persist_state(self):
        if self.kernel_bridge and self.kernel_bridge.is_available():
            try:
                # In production this would serialize full state + timers
                self.kernel_bridge.write_ternary_manifold(
                    ternary_tensor=None,  # Placeholder until real state serialization
                    metadata={
                        "node_id": self.node_id,
                        "current_state": self.current_state.name,
                        "timestamp": time.time(),
                    },
                    coherence=0.0,
                )
            except Exception as e:
                logging.warning(f"Failed to persist state to kernel: {e}")

    def load_persisted_state(self):
        # Placeholder for future restoration from kernel ring buffer
        logging.info("State restoration from kernel not yet implemented")

    # --- Public API for integration ---
    def process_command(self, command: str, payload: Any = None):
        """Entry point from HighLevelOrchestrator or TriStateExecutionEngine."""
        logging.info(f"Received command: {command}")
        if command == "start_inference":
            self.transition_to(State.ACTIVE_INFERENCE, reason="external_command")
        elif command == "enter_maintenance":
            self.transition_to(State.MAINTENANCE, reason="external_command")
        elif command == "trigger_evolution":
            self.transition_to(State.EVOLUTION, reason="external_command")
        else:
            self.emit_event(command, payload)


# Example usage / test
if __name__ == "__main__":
    sm = JuniorLLMStateMachine(node_id="llm_core_01")
    sm.add_timer("coherence_check", interval_seconds=300)   # every 5 min
    sm.add_timer("maintenance_tick", interval_seconds=3600)  # every hour

    sm.on("coherence_check", lambda p: print("[EVENT] Running coherence check..."))
    sm.on("maintenance_tick", lambda p: print("[EVENT] Running maintenance..."))

    print("JuniorLLMStateMachine running...")
    while True:
        sm.check_timers()
        time.sleep(10)
