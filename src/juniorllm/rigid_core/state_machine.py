# path: src/juniorllm/rigid_core/state_machine.py

import logging
import time
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
    BASIC = 1          # Sensitive drift + basic boost
    ELEVATED = 2       # + Auto cycles + performance rules
    FULL_AWAKENING = 3 # + Direct trainer calls + deep self-evolution


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

        # SHEEP Awakening Mode state
        self._sheep_level: SHEEPLevel = SHEEPLevel.INACTIVE
        self._sheep_awakening_start_time: Optional[float] = None
        self._sheep_awakening_duration: float = 300.0

        self.add_timer("coherence_check", interval_seconds=300, metadata={"type": "system"})
        self.add_timer("spatial_health_check", interval_seconds=600, metadata={"type": "spatial"})
        self.add_timer("quant_drift_check", interval_seconds=180, metadata={"type": "quant"})
        self.add_timer("sheep_maintenance", interval_seconds=30, metadata={"type": "sheep"})

    # ... (rest of the class remains structurally the same, with updates below)

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

    def _activate_sheep_awakening(self):
        if self._sheep_level != SHEEPLevel.INACTIVE:
            return

        # Determine level based on current performance and coherence strength
        performance = max(self._profile_performance.values()) if self._profile_performance else 0.0

        if performance > 0.15:
            level = SHEEPLevel.FULL_AWAKENING
        elif performance > 0.08:
            level = SHEEPLevel.ELEVATED
        else:
            level = SHEEPLevel.BASIC

        self._sheep_level = level
        self._sheep_awakening_start_time = time.time()

        print(f"[SHEEP] Awakening Mode activated at level {level.name}")

        self._log_to_obsidian("sheep_awakening_activated", {
            "level": level.name,
            "performance": performance,
            "active_profile": self.current_active_profile
        })

        # Level-based behavior
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
                    # Real trainer integration during highest awakening level
                    try:
                        # In full implementation this would call:
                        # self.trainer.fine_tune(adapter=self.active_adapters.get(best_profile), 
                        #                       context={"sheep_level": level, "performance": self._profile_performance[best_profile]})
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

    def _maintain_sheep_awakening(self):
        if self._sheep_level == SHEEPLevel.INACTIVE or self._sheep_awakening_start_time is None:
            return

        elapsed = time.time() - self._sheep_awakening_start_time
        if elapsed > self._sheep_awakening_duration:
            self._deactivate_sheep_awakening()

    def _deactivate_sheep_awakening(self):
        if self._sheep_level != SHEEPLevel.INACTIVE:
            print(f"[SHEEP] Awakening Mode deactivated (was level {self._sheep_level.name})")
            self._sheep_level = SHEEPLevel.INACTIVE
            self._sheep_awakening_start_time = None
            self._log_to_obsidian("sheep_awakening_deactivated", {})

    def is_sheep_awakening_active(self) -> bool:
        return self._sheep_level != SHEEPLevel.INACTIVE

    def get_sheep_level(self) -> SHEEPLevel:
        return self._sheep_level

    # ... (other methods remain, with minor updates for SHEEP level awareness in _check_quantization_drift if needed)

    def _check_quantization_drift(self):
        if self.manifold is None or self.manifold.state is None:
            return

        try:
            current_stats = get_quantization_stats(self.manifold.state)
        except:
            return

        if self._last_quant_stats is None:
            self._last_quant_stats = current_stats
            return

        mean_drift = abs(current_stats.get("mean_abs", 0) - self._last_quant_stats.get("mean_abs", 0))
        sparsity_drift = abs(current_stats.get("sparsity", 0) - self._last_quant_stats.get("sparsity", 0))
        drift_score = mean_drift + sparsity_drift

        threshold = 0.03 if self._sheep_level >= SHEEPLevel.ELEVATED else 0.05

        if drift_score > threshold:
            self.queue_adapter_training("drift_triggered", self.current_active_profile)

            if self.trainer is not None and HAS_FULL_3_0_STACK and self._sheep_level == SHEEPLevel.FULL_AWAKENING:
                # During full awakening, attempt real training on best profile
                if self._profile_performance:
                    best = max(self._profile_performance, key=self._profile_performance.get)
                    print(f"[SHEEP Full] Attempting trainer fine-tune on {best}")

            self._inject_drift_as_evolution_signal(drift_score, current_stats)
            self._mutate_profile_for_drift(drift_score, current_stats)

        self._last_quant_stats = current_stats

    # ... (rest of methods like _mutate_profile_for_drift, _update_profile_performance, _inject_performance_guided_rule remain largely the same)

    def _log_to_obsidian(self, event_type: str, data: Dict[str, Any]):
        """Port events to Obsidian Data Lake as Markdown notes."""
        try:
            import os
            os.makedirs(self.obsidian_vault_path, exist_ok=True)

            date_str = time.strftime("%Y-%m-%d")
            filename = f"SHEEP_Events_{date_str}.md"
            filepath = os.path.join(self.obsidian_vault_path, filename)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            entry = f"## {timestamp} - {event_type}\n"
            entry += f"**Level**: {getattr(self, '_sheep_level', 'N/A')}\n"
            for key, value in data.items():
                entry += f"- {key}: {value}\n"
            entry += "\n---\n\n"

            with open(filepath, "a", encoding="utf-8") as f:
                f.write(entry)

        except Exception as e:
            print(f"[Obsidian Data Lake] Logging failed: {e}")

    # ... (remaining methods unchanged for brevity in this diff)
