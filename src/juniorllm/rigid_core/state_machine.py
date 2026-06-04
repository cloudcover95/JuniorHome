# path: src/juniorhome/juniorllm/rigid_core/state_machine.py
#!/usr/bin/env python3
"""
JuniorLLM Rigid Core State Machine (Hierarchical + Capital Focus)

Added:
- Hierarchical sub-state support (nested states under CAPITAL_RECOVERY)
- Expanded capital accumulation logic with recovery sub-flows
- Better structure for long-term timers and persistence

This continues building the rigid Layer 1 core with real hierarchical behavior.
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
    CAPITAL_MONITORING = auto()
    CAPITAL_RECOVERY = auto()


# Hierarchical sub-states under CAPITAL_RECOVERY
class RecoverySubState(Enum):
    CLEAR_BLOCKS = auto()
    RESTART_PIPELINE = auto()
    VERIFY_ANCHOR = auto()
    ACH_REROUTE = auto()
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
        self.current_sub_state: Optional[RecoverySubState] = None
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL_BRIDGE else None

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
            self.current_sub_state = None  # Reset sub-state on root transition
            self._persist_state()

    def transition_sub_state(self, new_sub_state: RecoverySubState, reason: str = ""):
        if self.current_state != State.CAPITAL_RECOVERY:
            self.transition_to(State.CAPITAL_RECOVERY, reason="sub_state_entry")

        if new_sub_state != self.current_sub_state:
            self.history.append({
                "from_sub": self.current_sub_state.name if self.current_sub_state else None,
                "to_sub": new_sub_state.name,
                "timestamp": time.time(),
                "reason": reason
            })
            logging.info(f"Sub-state transition: {self.current_sub_state} -> {new_sub_state} ({reason})")
            self.current_sub_state = new_sub_state
            self._persist_state()

    def get_current_state(self) -> State:
        return self.current_state

    def get_current_sub_state(self) -> Optional[RecoverySubState]:
        return self.current_sub_state

    # --- Timer Management ---
    def add_timer(self, name: str, interval_seconds: float, persistent: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.timers[name] = Timer(name=name, interval=interval_seconds, persistent=persistent, metadata=metadata or {})

    def check_timers(self):
        for timer in list(self.timers.values()):
            if timer.should_fire():
                timer.fire()
                self._handle_timer_event(timer)

    def _handle_timer_event(self, timer: Timer):
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
                        "current_sub_state": self.current_sub_state.name if self.current_sub_state else None,
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
        elif command == "monitor_capital":
            self.transition_to(State.CAPITAL_MONITORING, reason="external_command")
        elif command == "recover_capital_pipeline":
            self.transition_to(State.CAPITAL_RECOVERY, reason="external_command")
            self.transition_sub_state(RecoverySubState.CLEAR_BLOCKS, reason="recovery_start")
        else:
            self.emit_event(command, payload)

    def handle_capital_task(self, task_type: str):
        if task_type == "capital_accumulation_monitor":
            self.transition_to(State.CAPITAL_MONITORING, reason="task")
            return {"status": "monitoring_started"}

        elif task_type == "capital_accumulation_restart":
            self.transition_to(State.CAPITAL_RECOVERY, reason="task")
            self.transition_sub_state(RecoverySubState.CLEAR_BLOCKS)
            return {"status": "recovery_started"}

        elif task_type == "verify_fiat_anchor":
            if self.current_state == State.CAPITAL_RECOVERY:
                self.transition_sub_state(RecoverySubState.VERIFY_ANCHOR)
            else:
                self.transition_to(State.MAINTENANCE, reason="task")
            return {"status": "verification_started"}

        elif task_type == "clear_external_blocks":
            if self.current_state == State.CAPITAL_RECOVERY:
                self.transition_sub_state(RecoverySubState.CLEAR_BLOCKS)
            return {"status": "clearing_blocks"}

        elif task_type == "ach_reroute":
            if self.current_state == State.CAPITAL_RECOVERY:
                self.transition_sub_state(RecoverySubState.ACH_REROUTE)
            return {"status": "initiating_ach_reroute"}

        return {"status": "unknown_task"}


if __name__ == "__main__":
    sm = JuniorLLMStateMachine(node_id="llm_core_01")
    sm.on("capital_health_check", lambda p: print("[EVENT] Checking capital pipeline..."))

    print("JuniorLLMStateMachine with hierarchical capital recovery running...")
    while True:
        sm.check_timers()
        time.sleep(5)
