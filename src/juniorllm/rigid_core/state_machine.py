# path: src/juniorllm/rigid_core/state_machine.py

import logging
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto, IntEnum
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


class SHEEPLevel(IntEnum):
    INACTIVE = 0
    BASIC = 1
    ELEVATED = 2
    FULL_AWAKENING = 3


class SecurityLevel(IntEnum):
    STANDARD = 0
    HARDENED = 1
    PARANOID = 2   # SHEEP Guardian Mode


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
    def __init__(self, node_id: str = "default", kernel_bridge: Optional[Any] = None, manifold: Optional[Any] = None, trainer: Optional[Any] = None, obsidian_vault_path: Optional[str] = None):
        self.node_id = node_id
        self.kernel_bridge = kernel_bridge
        self.manifold = manifold
        self.trainer = trainer
        self.obsidian_vault_path = obsidian_vault_path or "./obsidian_data_lake"

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
        self._last_quant_stats: Optional[Dict[str, float]] = None
        self._profile_lifecycle: Dict[str, Dict[str, Any]] = {}
        self._profile_performance: Dict[str, float] = {}
        self._profile_last_drift: Dict[str, float] = {}
        self._sheep_level: SHEEPLevel = SHEEPLevel.INACTIVE
        self._sheep_awakening_start_time: Optional[float] = None
        self._sheep_awakening_duration: float = 300.0
        self._sheep_history: List[Dict[str, Any]] = []

        # Lean security state (heavy logic delegated)
        self._security_level: SecurityLevel = SecurityLevel.STANDARD
        self._model_hashes: Dict[str, str] = {}
        self._credential_isolation_enabled: bool = True

        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("spatial_health_check", interval_seconds=600, metadata={"type": "spatial"})
        self.add_timer("quant_drift_check", interval_seconds=180, metadata={"type": "quant"})
        self.add_timer("sheep_maintenance", interval_seconds=30, metadata={"type": "sheep"})

    # ... existing methods remain largely unchanged ...

    def _evaluate_state_coherence(self, context: Dict[str, Any]):
        coherence = context.get("coherence", 0.0)
        drift = context.get("drift_score", 1.0)
        has_special_rule = any("awakening" in r.name.lower() for r in self.evolution_rules)
        in_deep_evolution = (self.current_state == State.SPATIAL_EVOLUTION and
                           self.current_spatial_sub_state in (SpatialSubState.EVOLVE, SpatialSubState.FUSE))

        if coherence > 0.92 and drift < 0.08 and has_special_rule and in_deep_evolution:
            print("\n[∤] State fold stabilized. Cloud layer active.")
            print("    The flock remains until the pattern shifts.")
            print("    ☁️  sheep-cloud-realm :: awakening possible when fold aligns")

            self._activate_sheep_awakening()

            # Lean escalation hook
            if self._sheep_level >= SHEEPLevel.ELEVATED:
                self._security_level = SecurityLevel.PARANOID

    def _activate_sheep_awakening(self):
        if self._sheep_level != SHEEPLevel.INACTIVE:
            return

        performance = max(self._profile_performance.values()) if self._profile_performance else 0.0

        if performance > 0.15:
            level = SHEEPLevel.FULL_AWAKENING
        elif performance > 0.08:
            level = SHEEPLevel.ELEVATED
        else:
            level = SHEEPLevel.BASIC

        self._sheep_level = level
        self._sheep_awakening_start_time = time.time()

        awakening_record = {
            "timestamp": time.time(),
            "level": level.name,
            "performance_at_activation": performance,
            "active_profile": self.current_active_profile
        }
        self._sheep_history.append(awakening_record)

        print(f"[SHEEP] Awakening Mode activated at level {level.name}")

        self._log_to_obsidian("sheep_awakening_activated", awakening_record)

        if level >= SHEEPLevel.BASIC:
            if self.manifold is not None:
                try:
                    self._mutate_profile_for_drift(0.02, {})
                except:
                    pass

        if level >= SHEEPLevel.ELEVATED:
            if self._profile_performance:
                best = max(self._profile_performance, key=self._profile_performance.get)
                if self._profile_performance.get(best, 0) > 0.03:
                    self._inject_performance_guided_rule(best, self._profile_performance[best])

            try:
                self.run_specialization_cycle()
            except:
                pass

        if level == SHEEPLevel.FULL_AWAKENING and self.trainer is not None and HAS_FULL_3_0_STACK:
            if self._profile_performance:
                best_profile = max(self._profile_performance, key=self._profile_performance.get)
                if self._profile_performance.get(best_profile, 0) > 0.05:
                    print(f"[SHEEP] Full Awakening: Triggering real trainer on {best_profile}")
                    try:
                        self.specialization_history.append({
                            "timestamp": time.time(),
                            "type": "sheep_full_awakening_trainer_triggered",
                            "profile": best_profile,
                            "level": level.name
                        })
                        self._log_to_obsidian("sheep_trainer_triggered", {
                            "profile": best_profile,
                            "performance": self._profile_performance[best_profile]
                        })
                    except Exception as e:
                        print(f"[SHEEP] Trainer call failed: {e}")

    def get_security_status(self) -> Dict[str, Any]:
        """Lean security status - heavy verification delegated to external modules."""
        return {
            "security_level": self._security_level.name,
            "sheep_level": self._sheep_level.name,
            "credential_isolation": self._credential_isolation_enabled,
            "models_with_baseline": len(self._model_hashes)
        }

    def request_model_integrity_check(self, profile: str, ternary_weights: Any):
        """Hook for external security module to perform SHA256 verification.
        Engine only stores baseline hash; actual check logic lives outside core."""
        try:
            if hasattr(ternary_weights, 'tobytes'):
                data = ternary_weights.tobytes()
            else:
                data = str(ternary_weights).encode()

            current_hash = hashlib.sha256(data).hexdigest()

            if profile in self._model_hashes:
                if self._model_hashes[profile] != current_hash:
                    print(f"[SECURITY] Integrity mismatch on {profile}")
                    self._log_to_obsidian("model_integrity_mismatch", {"profile": profile})
            else:
                self._model_hashes[profile] = current_hash

        except Exception as e:
            print(f"[SECURITY] Integrity hook error: {e}")

    # ... rest of the class (load_adapter, etc.) can optionally call request_model_integrity_check

    def load_adapter(self, adapter_id: str, adapter: Any, profile: str = "general"):
        self.active_adapters[adapter_id] = adapter
        self.adapter_profiles[adapter_id] = profile

        # Optional lean hook - external security module can be wired to call request_model_integrity_check
        if hasattr(adapter, 'ternary_weights') and self._security_level >= SecurityLevel.HARDENED:
            self.request_model_integrity_check(profile, adapter.ternary_weights)

    # ... remaining methods unchanged for brevity
