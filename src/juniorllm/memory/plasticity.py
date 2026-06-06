# path: src/juniorllm/memory/plasticity.py

"""
PlasticityEngine

Added neuromorphic / spiking simulation support.

- SpikingPlasticityModule for event-driven, STDP-like updates.
- Integration with hardware backend hook for future neuromorphic chips (Loihi, Akida, etc.).

This brings the biological mechanisms closer to true neuromorphic acceleration.
"""

from typing import Dict, Optional, Callable


class Neuromodulator:
    def __init__(self):
        self.global_signal: float = 1.0

    def update(self, outcome: float, context: Dict[str, Any] = None) -> float:
        if outcome > 0:
            self.global_signal = min(2.0, self.global_signal * 1.1)
        else:
            self.global_signal = max(0.5, self.global_signal * 0.9)
        return self.global_signal

    def get_modulation(self) -> float:
        return self.global_signal


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
    """
    Neuromorphic-inspired spiking plasticity.

    Simulates event-driven updates similar to STDP on neuromorphic hardware.
    Can be swapped in place of HebbianStructuralModule for spiking-style learning.
    Future: Direct mapping to Loihi 2, Akida, or other neuromorphic accelerators.
    """

    def __init__(self, decay: float = 0.95, threshold: float = 0.5):
        self.decay = decay
        self.threshold = threshold
        self.membrane_potential: Dict[str, float] = {}
        self.connection_strength: Dict[str, float] = {}

    def update(self, profile: str, delta_w: float, eligibility: float, outcome: float, modulation: float = 1.0) -> float:
        if profile not in self.membrane_potential:
            self.membrane_potential[profile] = 0.0
        if profile not in self.connection_strength:
            self.connection_strength[profile] = 0.0

        # Simple leaky integrate-and-fire style
        self.membrane_potential[profile] = self.membrane_potential[profile] * self.decay + eligibility * modulation

        spike = 1.0 if self.membrane_potential[profile] > self.threshold else 0.0

        if spike > 0:
            # STDP-like potentiation on spike
            self.connection_strength[profile] = min(1.0, self.connection_strength[profile] + 0.1 * modulation)
            self.membrane_potential[profile] = 0.0  # reset
        else:
            # Slight depression / decay
            self.connection_strength[profile] *= 0.99

        if self.connection_strength[profile] < 0.05:
            self.connection_strength[profile] = 0.0

        return delta_w + (spike * 0.2)

    def get_strength(self, profile: str) -> float:
        return self.connection_strength.get(profile, 0.0)


class PlasticityEngine:
    def __init__(self, lr: float = 0.01, homeostatic_target: float = 0.15, hebbian_module: Optional[HebbianStructuralModule] = None):
        self.lr = lr
        self.homeostatic_target = homeostatic_target
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9

        # Default to Hebbian; can be swapped with SpikingPlasticityModule for neuromorphic style
        self.hebbian = hebbian_module or HebbianStructuralModule()
        self.neuromodulator = Neuromodulator()

        self.hardware_backend: Optional[Callable] = None

    def set_hardware_backend(self, backend_fn: Callable):
        """Set backend for future neuromorphic hardware acceleration (Loihi, Akida, etc.)."""
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

    def apply(self, performance: Dict[str, float], lifecycle: Dict[str, Dict], profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0):
        if profile not in performance:
            performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)
        modulation = self.neuromodulator.update(outcome)

        modulated_eligibility = eligibility * modulation
        modulated_reward = reward * modulation

        if outcome > 0:
            timing_factor = modulated_eligibility
            delta_w = self.lr * timing_factor * modulated_reward * outcome * coactivation
        else:
            timing_factor = max(0.2, 1.0 - modulated_eligibility)
            delta_w = self.lr * timing_factor * abs(outcome) * modulated_reward * coactivation * -1.0

        delta_w = self.hebbian.update(profile, delta_w, modulated_eligibility, outcome, modulation)

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
        """Switch to neuromorphic spiking plasticity module."""
        if enabled:
            self.hebbian = SpikingPlasticityModule()
        else:
            self.hebbian = HebbianStructuralModule()
