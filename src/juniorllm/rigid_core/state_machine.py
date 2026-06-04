# path: src/juniorhome/juniorllm/rigid_core/state_machine.py
#!/usr/bin/env python3
"""
JuniorLLM Rigid Core State Machine (Expanded - Evolution Rules + Security Policies)

Major updates:
- Evolution rule system with pluggable rules
- Guard condition evaluation with security policy hooks
- Deeper isolated persistence (state + rules + spatial metadata)
- Stronger black-box boundaries for privacy and security

Designed for secure, long-term autonomous operation with controllable boundaries.
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
    SPATIAL_MONITORING = auto()
    SPATIAL_EVOLUTION = auto()


class SpatialSubState(Enum):
    INITIALIZE = auto()
    TRACK = auto()
    DETECT_DRIFT = auto()
    EVOLVE = auto()
    FUSE = auto()
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


@dataclass
class EvolutionRule:
    """Simple evolution rule with optional security policy hook."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]   # Receives context (coherence, drift, etc.)
    action: Callable[[], None]                    # What to do when rule triggers
    security_policy: Optional[str] = None         # e.g. "require_auth", "rate_limit", "anomaly_check"


class JuniorLLMStateMachine:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.current_state: State = State.IDLE
        self.current_spatial_sub_state: Optional[SpatialSubState] = None
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.evolution_rules: List[EvolutionRule] = []
        self.kernel_bridge = JuniorOSKernelBridge() if HAS_KERNEL_BRIDGE else None

        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("spatial_health_check", interval_seconds=600, metadata={"type": "spatial"})

        logging.info(f"JuniorLLMStateMachine initialized for node {node_id}")

    def add_evolution_rule(self, rule: EvolutionRule):
        self.evolution_rules.append(rule)
        logging.info(f"Evolution rule added: {rule.name} (security_policy={rule.security_policy})")

    def evaluate_evolution_rules(self, context: Dict[str, Any]):
        """Evaluate all rules. Guard conditions + security policies can block actions."""
        for rule in self.evolution_rules:
            if rule.condition(context):
                # Basic security policy hook (expandable)
                if rule.security_policy == "require_auth":
                    if not context.get("authenticated", False):
                        logging.warning(f"Rule {rule.name} blocked: authentication required")
                        continue
                if rule.security_policy == "anomaly_check":
                    if context.get("anomaly_score", 0) > 0.8:
                        logging.warning(f"Rule {rule.name} blocked: high anomaly score")
                        continue

                logging.info(f"Evolution rule triggered: {rule.name}")
                rule.action()

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

    # --- Persistence (Deeper + Isolated) ---
    def _persist_state(self):
        if self.kernel_bridge and self.kernel_bridge.is_available():
            try:
                self.kernel_bridge.write_ternary_manifold(
                    ternary_tensor=None,
                    metadata={
                        "node_id": self.node_id,
                        "current_state": self.current_state.name,
                        "current_spatial_sub_state": self.current_spatial_sub_state.name if self.current_spatial_sub_state else None,
                        "active_evolution_rules": [r.name for r in self.evolution_rules],
                        "history_length": len(self.history),
                        "timestamp": time.time(),
                    },
                    coherence=0.0,
                )
            except Exception as e:
                logging.warning(f"Failed to persist state: {e}")

    # --- Public API (Black-box friendly) ---
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

    # Example evolution rule with security policy
    def high_drift_rule(context):
        return context.get("drift_score", 0) > 0.7

    def trigger_evolution():
        print("[RULE] High drift detected - triggering evolution")

    sm.add_evolution_rule(EvolutionRule(
        name="high_drift_evolution",
        condition=high_drift_rule,
        action=trigger_evolution,
        security_policy="anomaly_check"
    ))

    print("JuniorLLMStateMachine with evolution rules + security policies running...")
    while True:
        sm.check_timers()
        sm.evaluate_evolution_rules({"drift_score": 0.85, "anomaly_score": 0.3})
        time.sleep(5)
