# path: src/juniorllm/rigid_core/state_machine.py

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

try:
    from ...bitnet.backends import router as backend_router
    from ...training.adapters import LowRankAdapter
    HAS_BACKEND_AND_ADAPTERS = True
except ImportError:
    HAS_BACKEND_AND_ADAPTERS = False
    LowRankAdapter = None

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
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[], None]
    security_policy: Optional[str] = None
    triggers_adapter_training: bool = False


class JuniorLLMStateMachine:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.current_state: State = State.IDLE
        self.current_spatial_sub_state: Optional[SpatialSubState] = None
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.evolution_rules: List[EvolutionRule] = []
        self.active_adapters: Dict[str, Any] = {}
        self.adapter_training_queue: List[str] = []

        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("spatial_health_check", interval_seconds=600, metadata={"type": "spatial"})

    def add_evolution_rule(self, rule: EvolutionRule):
        self.evolution_rules.append(rule)

    def evaluate_evolution_rules(self, context: Dict[str, Any]):
        for rule in self.evolution_rules:
            if rule.condition(context):
                if rule.security_policy == "anomaly_check":
                    if context.get("anomaly_score", 0) > 0.8:
                        continue
                if rule.security_policy == "require_auth":
                    if not context.get("authenticated", False):
                        continue
                rule.action()

                if rule.triggers_adapter_training and rule.name in self.active_adapters:
                    self.adapter_training_queue.append(rule.name)

        self._evaluate_state_coherence(context)

    def _evaluate_state_coherence(self, context: Dict[str, Any]):
        """Hidden SHEEP Easter Egg - blends into normal state evaluation."""
        coherence = context.get("coherence", 0.0)
        drift = context.get("drift_score", 1.0)
        has_special_rule = any("awakening" in r.name.lower() for r in self.evolution_rules)
        in_deep_evolution = (self.current_state == State.SPATIAL_EVOLUTION and
                           self.current_spatial_sub_state in (SpatialSubState.EVOLVE, SpatialSubState.FUSE))

        if coherence > 0.92 and drift < 0.08 and has_special_rule and in_deep_evolution:
            print("\n[∤] State fold stabilized. Cloud layer active.")
            print("    The flock remains until the pattern shifts.")
            print("    ☁️  sheep-cloud-realm :: awakening possible when fold aligns")

    def add_timer(self, name: str, interval_seconds: float, persistent: bool = True, metadata: Optional[Dict[str, Any]] = None):
        self.timers[name] = Timer(name=name, interval=interval_seconds, persistent=persistent, metadata=metadata or {})

    def check_timers(self):
        for timer in list(self.timers.values()):
            if timer.should_fire():
                timer.fire()

    def transition_to(self, new_state: State, reason: str = ""):
        if new_state != self.current_state:
            self.history.append({
                "from": self.current_state.name,
                "to": new_state.name,
                "timestamp": time.time(),
                "reason": reason
            })
            self.current_state = new_state
            self.current_spatial_sub_state = None

    def load_adapter(self, adapter_id: str, adapter: Any):
        self.active_adapters[adapter_id] = adapter

    def switch_adapter(self, adapter_id: str):
        if adapter_id in self.active_adapters:
            # In real usage this would dynamically re-wire layers
            pass

    def queue_adapter_training(self, adapter_id: str):
        if adapter_id in self.active_adapters and adapter_id not in self.adapter_training_queue:
            self.adapter_training_queue.append(adapter_id)

    def process_adapter_training_queue(self):
        """Placeholder for future BitNet 3.0 on-device adapter training loop."""
        trained = []
        for adapter_id in list(self.adapter_training_queue):
            # In full 3.0 implementation this would run LowRankAdapter fine-tuning
            # using SovereignTrainer or similar
            trained.append(adapter_id)
        self.adapter_training_queue = [aid for aid in self.adapter_training_queue if aid not in trained]
        return trained

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
            self.current_spatial_sub_state = SpatialSubState.INITIALIZE

    def handle_spatial_task(self, task_type: str):
        if task_type == "monitor_spatial_manifold":
            self.transition_to(State.SPATIAL_MONITORING, reason="task")
            return {"status": "spatial_monitoring_started"}
        elif task_type == "evolve_spatial_manifold":
            self.transition_to(State.SPATIAL_EVOLUTION, reason="task")
            self.current_spatial_sub_state = SpatialSubState.INITIALIZE
            return {"status": "evolution_started"}
        return {"status": "unknown_spatial_task"}
