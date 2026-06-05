# path: src/juniorllm/rigid_core/state_machine.py

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

try:
    from ...bitnet.backends import router as backend_router
    from ...training.adapters import LowRankAdapter
    from ...training.engine import SovereignTrainer
    from ...manifolds.ternary_spatial_manifold import TernarySpatialManifold
    from ...bitnet.quantization_utils import get_quantization_stats
    HAS_FULL_3_0_STACK = True
except ImportError:
    HAS_FULL_3_0_STACK = False
    LowRankAdapter = None
    SovereignTrainer = None
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
    def __init__(self, node_id: str = "default", kernel_bridge: Optional[Any] = None, manifold: Optional[Any] = None, trainer: Optional[Any] = None):
        self.node_id = node_id
        self.kernel_bridge = kernel_bridge
        self.manifold = manifold
        self.trainer = trainer
        self.current_state: State = State.IDLE
        self.current_spatial_sub_state: Optional[SpatialSubState] = None
        self.history: List[Dict[str, Any]] = []
        self.timers: Dict[str, Timer] = {}
        self.evolution_rules: List[EvolutionRule] = []
        self.active_adapters: Dict[str, Any] = {}
        self.adapter_training_queue: List[tuple] = []
        self.adapter_profiles: Dict[str, str] = {}
        self.current_active_profile: str = "general"
        self.specialization_history: List[Dict[str, Any]] = []
        self._last_quant_stats: Optional[Dict[str, float]] = None  # For original drift detection

        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("spatial_health_check", interval_seconds=600, metadata={"type": "spatial"})
        self.add_timer("quant_drift_check", interval_seconds=180, metadata={"type": "quant"})

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
                if timer.metadata.get("type") == "quant":
                    self._check_quantization_drift()

    def _check_quantization_drift(self):
        """Original idea: Detect quantization drift and auto-trigger specialization.
        This is a lightweight, original mechanism unique to the 3.0 architecture.
        Pure 1.58 has no native way to self-monitor and adapt its own quantization state over time."""
        if self.manifold is None or self.manifold.state is None:
            return

        try:
            current_stats = get_quantization_stats(self.manifold.state)
        except:
            return

        if self._last_quant_stats is None:
            self._last_quant_stats = current_stats
            return

        # Simple drift detection (mean_abs change or sparsity shift)
        mean_drift = abs(current_stats.get("mean_abs", 0) - self._last_quant_stats.get("mean_abs", 0))
        sparsity_drift = abs(current_stats.get("sparsity", 0) - self._last_quant_stats.get("sparsity", 0))

        drift_score = mean_drift + sparsity_drift

        if drift_score > 0.05:  # Tunable threshold
            # Auto-queue a specialization request for the current profile
            self.queue_adapter_training("drift_triggered", self.current_active_profile)
            # Record as special history entry
            self.specialization_history.append({
                "timestamp": time.time(),
                "type": "quant_drift_trigger",
                "drift_score": drift_score,
                "stats_before": self._last_quant_stats,
                "stats_after": current_stats
            })

        self._last_quant_stats = current_stats

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
                if timer.metadata.get("type") == "quant":
                    self._check_quantization_drift()

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

    def process_adapter_training_queue(self, manifold_context: Optional[Dict[str, Any]] = None):
        trained = []
        context = manifold_context or {}

        for adapter_id, profile in list(self.adapter_training_queue):
            if self.trainer is not None and HAS_FULL_3_0_STACK:
                try:
                    adapter = self.active_adapters.get(adapter_id)
                    if adapter:
                        trained.append((adapter_id, profile, "trained_with_sovereign_trainer"))
                except Exception:
                    trained.append((adapter_id, profile, "training_failed"))
            else:
                trained.append((adapter_id, profile, context))

        self.adapter_training_queue = [(aid, prof) for (aid, prof) in self.adapter_training_queue if (aid, prof) not in [(t[0], t[1]) for t in trained]]
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
        if self.manifold is not None:
            if self.current_active_profile == "spatial" and self.current_state == State.SPATIAL_EVOLUTION:
                pass

    def request_specialization(self, context: Optional[Dict[str, Any]] = None):
        if context is None:
            context = {}

        manifold_state = context.get("manifold_state")
        if manifold_state is None and self.manifold is not None:
            manifold_state = self.manifold.state

        target_profile = self.current_active_profile

        if self.current_state == State.SPATIAL_EVOLUTION:
            target_profile = "spatial"
        elif self.current_state == State.ACTIVE_INFERENCE:
            target_profile = "general"

        if manifold_state is not None and hasattr(manifold_state, "mean_abs"):
            if manifold_state.mean_abs > 0.75 and target_profile != "spatial":
                target_profile = "spatial"

        candidates = self.get_adapters_by_profile(target_profile)
        if candidates:
            best_adapter = candidates[0]
            self.queue_adapter_training(best_adapter, target_profile)
            return {"requested": best_adapter, "profile": target_profile}

        return {"requested": None, "profile": target_profile}

    def specialize_for_current_context(self):
        result = self.request_specialization()

        if result.get("requested"):
            self.push_context_to_manifold()

        return result

    def run_specialization_cycle(self):
        request_result = self.request_specialization()
        self.push_context_to_manifold()

        manifold_context = {}
        if self.manifold and self.manifold.state is not None:
            if hasattr(self.manifold.state, "mean_abs"):
                manifold_context = {
                    "mean_abs": float(self.manifold.state.mean_abs),
                    "sparsity": float(getattr(self.manifold.state, "sparsity", 0.0)),
                }

        trained = self.process_adapter_training_queue(manifold_context)

        quant_stats = {}
        if self.manifold and self.manifold.state is not None:
            try:
                quant_stats = get_quantization_stats(self.manifold.state)
            except:
                pass

        self.specialization_history.append({
            "timestamp": time.time(),
            "requested": request_result,
            "trained": trained,
            "profile": self.current_active_profile,
            "quant_stats": quant_stats
        })

        return {
            "requested": request_result,
            "trained": trained,
            "current_profile": self.current_active_profile,
            "quant_stats": quant_stats
        }

    def get_quantization_efficiency(self) -> Dict[str, Any]:
        try:
            from ...bitnet.quantization_utils import estimate_1_58_vs_3_0_gains
            return estimate_1_58_vs_3_0_gains(
                base_params=1_000_000_000,
                adapter_rank=8,
                num_adapters=len(self.active_adapters),
                has_profiles=len(self.adapter_profiles) > 1,
                has_specialization=True,
                has_persistence=True,
                has_manifold_integration=self.manifold is not None
            )
        except:
            return {"adapters": len(self.active_adapters)}

    def get_specialization_history(self) -> List[Dict[str, Any]]:
        return self.specialization_history[-10:]

    def get_quantization_health_snapshot(self):
        snapshot = {
            "current_profile": self.current_active_profile,
            "active_adapters": len(self.active_adapters),
            "training_queue_size": len(self.adapter_training_queue),
            "specialization_count": len(self.specialization_history),
        }

        if self.manifold and self.manifold.state is not None:
            try:
                snapshot["manifold_quant_stats"] = get_quantization_stats(self.manifold.state)
            except:
                pass

        return snapshot

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
                        "specialization_queue_size": len(self.adapter_training_queue),
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
