# path: src/juniorllm/rigid_core/state_machine.py

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

try:
    from ...bitnet.backends import router as backend_router
    from ...training.adapters import LowRankAdapter
    from ...manifolds.ternary_spatial_manifold import TernarySpatialManifold
    HAS_BACKEND_ADAPTERS_AND_MANIFOLD = True
except ImportError:
    HAS_BACKEND_ADAPTERS_AND_MANIFOLD = False
    LowRankAdapter = None
    TernarySpatialManifold = None

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
    last_triggered: float = field(default_factory=time.time())
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
    target_adapter_profile: Optional[str] = None


class JuniorLLMStateMachine:
    def __init__(self, node_id: str = "default", kernel_bridge: Optional[Any] = None, manifold: Optional[Any] = None):
        self.node_id = node_id
        self.kernel_bridge = kernel_bridge
        self.manifold = manifold
        self.current_state: State = State.IDLE
        self.current_spatial_sub_state: Optional[SpatialSubState] = None
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.evolution_rules: List[EvolutionRule] = []
        self.active_adapters: Dict[str, Any] = {}
        self.adapter_training_queue: List[tuple] = []
        self.adapter_profiles: Dict[str, str] = {}
        self.current_active_profile: str = "general"

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

                if rule.triggers_adapter_training and rule.target_adapter_profile:
                    self.queue_adapter_training(rule.name, rule.target_adapter_profile)

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

            if new_state == State.SPATIAL_EVOLUTION:
                self.switch_to_profile("spatial")
            elif new_state == State.ACTIVE_INFERENCE:
                self.switch_to_profile("general")

            self._persist_state()

    def load_adapter(self, adapter_id: str, adapter: Any, profile: str = "general"):
        self.active_adapters[adapter_id] = adapter
        self.adapter_profiles[adapter_id] = profile

    def switch_adapter(self, adapter_id: str):
        if adapter_id in self.active_adapters:
            self.current_active_profile = self.adapter_profiles.get(adapter_id, "general")

    def switch_to_profile(self, profile: str):
        self.current_active_profile = profile
        self._persist_state()

    def queue_adapter_training(self, adapter_id: str, profile: Optional[str] = None):
        if adapter_id in self.active_adapters:
            prof = profile or self.adapter_profiles.get(adapter_id, "general")
            entry = (adapter_id, prof)
            if entry not in self.adapter_training_queue:
                self.adapter_training_queue.append(entry)

    def process_adapter_training_queue(self):
        trained = []
        for adapter_id, profile in list(self.adapter_training_queue):
            trained.append((adapter_id, profile))
        self.adapter_training_queue = [(aid, prof) for (aid, prof) in self.adapter_training_queue if (aid, prof) not in trained]
        return trained

    def get_adapters_by_profile(self, profile: str) -> List[str]:
        return [aid for aid, prof in self.adapter_profiles.items() if prof == profile]

    def get_current_active_adapters(self) -> List[str]:
        return self.get_adapters_by_profile(self.current_active_profile)

    def update_from_manifold(self, manifold_state: Optional[Any] = None):
        if manifold_state is None and self.manifold is not None:
            manifold_state = self.manifold.state

        if manifold_state is not None:
            if hasattr(manifold_state, "mean_abs"):
                if manifold_state.mean_abs > 0.7:
                    if self.current_active_profile != "spatial":
                        self.switch_to_profile("spatial")

    def push_context_to_manifold(self):
        """Push current state/profile context back to the manifold for physics-informed updates."""
        if self.manifold is not None:
            # Example: when in spatial evolution, the manifold can receive priors from current profile
            if self.current_active_profile == "spatial" and self.current_state == State.SPATIAL_EVOLUTION:
                # In full implementation this would inject adapter-derived priors into manifold folding
                pass

    def _persist_state(self):
        if self.kernel_bridge and hasattr(self.kernel_bridge, "write_ternary_manifold"):
            try:
                manifold_features = {}
                if self.manifold and self.manifold.state is not None:
                    if hasattr(self.manifold.state, "mean_abs"):
                        manifold_features = {
                            "mean_abs": float(self.manifold.state.mean_abs),
                            "sparsity": float(getattr(self.manifold.state, "sparsity", 0.0)),
                        }

                self.kernel_bridge.write_ternary_manifold(
                    ternary_tensor=None,
                    metadata={
                        "node_id": self.node_id,
                        "current_state": self.current_state.name,
                        "current_active_profile": self.current_active_profile,
                        "adapter_profiles": self.adapter_profiles,
                        "manifold_features": manifold_features,
                        "timestamp": time.time(),
                    },
                    coherence=0.0,
                )
            except Exception:
                pass

    def restore_from_persistence(self, metadata: Dict[str, Any]):
        if "current_active_profile" in metadata:
            self.current_active_profile = metadata["current_active_profile"]
        if "adapter_profiles" in metadata:
            self.adapter_profiles.update(metadata["adapter_profiles"])

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
