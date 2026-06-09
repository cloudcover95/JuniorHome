# path: src/juniorllm/memory/plasticity.py

"""
PlasticityEngine

Deeper spiking / neuromorphic work:
- Enhanced SpikingPlasticityModule with better biological timing and homeostatic regulation.
- Richer training signal generation.
"""

from typing import Dict, Optional, Callable, Any, List
try:
    from src.quantization.hybrid_squeeze_bitnet import HybridSqueezeBitNetQuantizer
except ImportError:
    HybridSqueezeBitNetQuantizer = None


class Neuromodulator:
    def __init__(self):
        self.global_signal: float = 1.0
        self.history: List[float] = []

    def update(self, outcome: float, context: Dict[str, Any] = None) -> float:
        if outcome > 0:
            self.global_signal = min(2.0, self.global_signal * 1.1)
        else:
            self.global_signal = max(0.5, self.global_signal * 0.9)
        self.history.append(self.global_signal)
        if len(self.history) > 20:
            self.history.pop(0)
        return self.global_signal

    def get_modulation(self) -> float:
        return self.global_signal

    def get_trend(self) -> float:
        if len(self.history) < 3:
            return 0.0
        return self.history[-1] - self.history[0]


class HebbianStructuralModule:
    def __init__(self, growth_rate: float = 0.1, prune_threshold: float = 0.05):
        self.growth_rate = growth_rate
        self.prune_threshold = prune_threshold
        self.connection_strength: Dict[str, float] = {}

    def update(self, profile: str, delta_w: float, eligibility: float, outcome: float, modulation: float = 1.0) -> float:
        if profile not in self.connection_strength:
            self.connection_strength[profile] = 0.0

        effective_growth = self.growth_rate * modulation

        if outcome > 0 and eligibility > 0.3:
            self.connection_strength[profile] = min(
                1.0, self.connection_strength[profile] + effective_growth * eligibility
            )

        if self.connection_strength[profile] < self.prune_threshold:
            self.connection_strength[profile] = 0.0

        structural_boost = self.connection_strength[profile] * 0.5 if outcome > 0 else 0.0
        return delta_w + structural_boost

    def get_strength(self, profile: str) -> float:
        return self.connection_strength.get(profile, 0.0)


class SpikingPlasticityModule:
    """Deeper biological spiking model with improved timing and homeostasis."""
    def __init__(self, decay: float = 0.92, threshold: float = 0.55, stp_window: float = 0.25, homeostatic_target: float = 0.18):
        self.decay = decay
        self.threshold = threshold
        self.stp_window = stp_window
        self.homeostatic_target = homeostatic_target
        self.membrane_potential: Dict[str, float] = {}
        self.connection_strength: Dict[str, float] = {}
        self.last_spike_time: Dict[str, float] = {}
        self.spike_count: Dict[str, int] = {}

    def update(self, profile: str, delta_w: float, eligibility: float, outcome: float, modulation: float = 1.0, current_time: float = 0.0) -> float:
        if profile not in self.membrane_potential:
            self.membrane_potential[profile] = 0.0
            self.connection_strength[profile] = 0.0
            self.last_spike_time[profile] = 0.0
            self.spike_count[profile] = 0

        # Leaky integration with modulation
        self.membrane_potential[profile] = self.membrane_potential[profile] * self.decay + eligibility * modulation

        spike = 1.0 if self.membrane_potential[profile] > self.threshold else 0.0

        if spike > 0:
            time_since_last = current_time - self.last_spike_time.get(profile, 0.0)
            timing_factor = max(0.4, 1.0 - (time_since_last / self.stp_window)) if time_since_last < self.stp_window else 0.4

            self.connection_strength[profile] = min(1.0, self.connection_strength[profile] + 0.15 * modulation * timing_factor)
            self.membrane_potential[profile] = 0.0
            self.last_spike_time[profile] = current_time
            self.spike_count[profile] += 1
        else:
            self.connection_strength[profile] *= 0.98

        # Homeostatic regulation (more sophisticated)
        total_spikes = sum(self.spike_count.values())
        if total_spikes > 0:
            avg_rate = total_spikes / max(len(self.spike_count), 1)
            if avg_rate > self.homeostatic_target * 10:
                self.connection_strength[profile] *= 0.985

        if self.connection_strength[profile] < 0.03:
            self.connection_strength[profile] = 0.0

        return delta_w + (spike * 0.3 * timing_factor if 'timing_factor' in locals() else spike * 0.25)

    def get_strength(self, profile: str) -> float:
        return self.connection_strength.get(profile, 0.0)


class PlasticityEngine:
    def __init__(self, lr: float = 0.01, homeostatic_target: float = 0.15, hebbian_module: Optional[HebbianStructuralModule] = None):
        self.lr = lr
        self.homeostatic_target = homeostatic_target
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9

        self.hebbian = hebbian_module or HebbianStructuralModule()
        self.neuromodulator = Neuromodulator()
        self.hybrid_quantizer = HybridSqueezeBitNetQuantizer() if HybridSqueezeBitNetQuantizer else None

        self.hardware_backend: Optional[Callable] = None
        self.meta_plasticity_factor: float = 1.0
        self._last_update_time: float = 0.0

    def set_hardware_backend(self, backend_fn: Callable):
        self.hardware_backend = backend_fn

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        if profile not in self.eligibility_traces:
            self.eligibility_traces[profile] = 0.0

        modulation = self.neuromodulator.get_modulation()
        modulated_strength = strength * modulation
        self.eligibility_traces[profile] = min(1.0, self.eligibility_traces[profile] + modulated_strength)

    def decay_eligibility_traces(self):
        for profile in list(self.eligibility_traces.keys()):
            self.eligibility_traces[profile] *= self.eligibility_decay
            if self.eligibility_traces[profile] < 0.01:
                del self.eligibility_traces[profile]

    def apply(self, performance: Dict[str, float], lifecycle: Dict[str, Dict], profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0, current_time: float = None):
        if profile not in performance:
            performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)
        modulation = self.neuromodulator.update(outcome)

        modulated_eligibility = eligibility * modulation
        modulated_reward = reward * modulation

        effective_lr = self.lr * self.meta_plasticity_factor

        if outcome > 0:
            timing_factor = modulated_eligibility
            delta_w = effective_lr * timing_factor * modulated_reward * outcome * coactivation
        else:
            timing_factor = max(0.2, 1.0 - modulated_eligibility)
            delta_w = effective_lr * timing_factor * abs(outcome) * modulated_reward * coactivation * -1.0

        if current_time is None:
            current_time = self._last_update_time + 0.01
        self._last_update_time = current_time

        delta_w = self.hebbian.update(profile, delta_w, modulated_eligibility, outcome, modulation, current_time=current_time)

        if self.hybrid_quantizer:
            try:
                delta_w = float(self.hybrid_quantizer.quantize([delta_w])[0])
            except Exception:
                pass

        performance[profile] += delta_w

        if self.hebbian.get_strength(profile) > 0.7 and outcome > 0:
            performance[profile] *= 1.02

        current_avg = sum(performance.values()) / max(len(performance), 1)
        if current_avg > self.homeostatic_target:
            performance[profile] *= 0.995

        if profile in lifecycle:
            lifecycle[profile]["performance_score"] = performance[profile]
            lifecycle[profile]["connection_strength"] = self.hebbian.get_strength(profile)
            lifecycle[profile]["neuromodulation"] = modulation

        if profile in self.eligibility_traces:
            self.eligibility_traces[profile] *= 0.55

    def sleep_consolidation(self, active_profiles: list = None):
        for profile in list(self.hebbian.connection_strength.keys()):
            strength = self.hebbian.connection_strength[profile]
            if strength > 0.6:
                self.hebbian.connection_strength[profile] = min(1.0, strength * 1.05)
            elif strength < 0.2:
                self.hebbian.connection_strength[profile] = 0.0

    def get_connection_strength(self, profile: str) -> float:
        return self.hebbian.get_strength(profile)

    def get_neuromodulation(self) -> float:
        return self.neuromodulator.get_modulation()

    def verify_integrity(self) -> bool:
        return True

    def set_spiking_mode(self, enabled: bool = True):
        if enabled:
            self.hebbian = SpikingPlasticityModule()
        else:
            self.hebbian = HebbianStructuralModule()

    def adapt_meta_plasticity(self, recent_performance: float):
        if recent_performance > 0.8:
            self.meta_plasticity_factor = max(0.5, self.meta_plasticity_factor * 0.95)
        else:
            self.meta_plasticity_factor = min(2.0, self.meta_plasticity_factor * 1.05)

    def get_efficiency_report(self) -> Dict[str, Any]:
        return {
            "meta_plasticity_factor": self.meta_plasticity_factor,
            "neuromodulation_level": self.neuromodulator.get_modulation(),
            "active_profiles": len(self.eligibility_traces),
            "neuromod_trend": self.neuromodulator.get_trend()
        }

    def generate_training_signals(self, outcome: float, profile: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        modulation = self.neuromodulator.get_modulation()
        trend = self.neuromodulator.get_trend()
        eligibility = self.eligibility_traces.get(profile, 0.0)
        connection_strength = self.hebbian.get_strength(profile)

        confidence = min(1.0, max(0.0, (connection_strength + (trend * 0.5) + 0.5) / 2.0))

        signal = {
            "signal_id": f"{profile}_{int(time.time() * 1000)}",
            "profile": profile,
            "outcome": outcome,
            "modulation": modulation,
            "modulation_trend": trend,
            "eligibility": eligibility,
            "connection_strength": connection_strength,
            "confidence": confidence,
            "timestamp": time.time(),
            "context": context or {}
        }
        return signal

    def get_plasticity_training_data(self, max_items: int = 50) -> List[Dict[str, Any]]:
        return [{
            "type": "plasticity_training_signal",
            "note": "Use generate_training_signals() during apply() calls."
        }]
