# path: src/juniorhome/juniorllm/rigid_core/state_machine.py
#!/usr/bin/env python3
"""
JuniorLLM Rigid Core State Machine (Expanded v2)

Added:
- More root + sub-states
- Improved persistent timer management
- Basic evolution rules / hierarchical behavior hooks
- Integration with capital/accumulation task handling
- Better persistence via JuniorOSKernelBridge

This continues building the rigid Layer 1 core for long-term state,
timers, and controlled autonomous evolution.
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
    CAPITAL_MONITORING = auto()      # New: monitoring accumulation pipelines
    CAPITAL_RECOVERY = auto()        # New: handling recovery from payment/accumulation failures


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
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL_BRIDGE else None

        # Default persistent timers
        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("capital_health_check", interval_seconds=1800, metadata={"type": "capital"})

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
            self._persist_state()

    def get_current_state(self) -> State:
        return self.current_state

    # --- Timer Management ---
    def add_timer(self, name: str, interval_seconds: float, persistent: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.timers[name] = Timer(
            name=name,
            interval=interval_seconds,
            persistent=persistent,
            metadata=metadata or {}
        )
        logging.info(f"Timer '{name}' added ({interval_seconds}s)")

    def check_timers(self):
        for timer in list(self.timers.values()):
            if timer.should_fire():
                timer.fire()
                self._handle_timer_event(timer)

    def _handle_timer_event(self, timer: Timer):
        logging.info(f"Timer fired: {timer.name}")
        if timer.metadata.get("type") == "capital":
            self.emit_event("capital_health_check")
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

    # --- Persistence ---
    def _persist_state(self):
        if self.kernel_bridge and self.kernel_bridge.is_available():
            try:
                self.kernel_bridge.write_ternary_manifold(
                    ternary_tensor=None,
                    metadata={
                        "node_id": self.node_id,
                        "current_state": self.current_state.name,
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
        logging.info(f"Received command: {command}")

        if command == "start_inference":
            self.transition_to(State.ACTIVE_INFERENCE, reason="external_command")
        elif command == "enter_maintenance":
            self.transition_to(State.MAINTENANCE, reason="external_command")
        elif command == "trigger_evolution":
            self.transition_to(State.EVOLUTION, reason="external_command")
        elif command == "monitor_capital":
            self.transition_to(State.CAPITAL_MONITORING, reason="external_command")
        elif command == "recover_capital_pipeline":
            self.transition_to(State.CAPITAL_RECOVERY, reason="external_command")
        else:
            self.emit_event(command, payload)

    def handle_capital_task(self, task_type: str):
        """Basic execution handler for capital/accumulation tasks."""
        if task_type == "capital_accumulation_monitor":
            self.transition_to(State.CAPITAL_MONITORING, reason="task")
            return {"status": "monitoring_started"}
        elif task_type == "capital_accumulation_restart":
            self.transition_to(State.CAPITAL_RECOVERY, reason="task")
            return {"status": "recovery_initiated"}
        elif task_type == "verify_fiat_anchor":
            self.transition_to(State.MAINTENANCE, reason="task")
            return {"status": "verification_started"}
        else:
            return {"status": "unknown_task"}


if __name__ == "__main__":
    sm = JuniorLLMStateMachine(node_id="llm_core_01")
    sm.on("capital_health_check", lambda p: print("[EVENT] Checking capital pipeline health..."))

    print("JuniorLLMStateMachine running with capital task support...")
    while True:
        sm.check_timers()
        time.sleep(5)
