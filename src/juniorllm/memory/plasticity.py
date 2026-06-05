# path: src/juniorllm/memory/plasticity.py

"""
PlasticityEngine

Composable component for biologically-inspired plasticity rules.
Contains eligibility traces, reward modulation, Hebbian updates,
and homeostatic scaling.

Can be used independently or composed inside SHEEPMemory.
"""

from typing import Dict


class PlasticityEngine:
    def __init__(self, lr: float = 0.01, homeostatic_target: float = 0.15):
        self.lr = lr
        self.homeostatic_target = homeostatic_target
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        if profile not in self.eligibility_traces:
            self.eligibility_traces[profile] = 0.0
        self.eligibility_traces[profile] = min(1.0, self.eligibility_traces[profile] + strength)

    def decay_eligibility_traces(self):
        for profile in list(self.eligibility_traces.keys()):
            self.eligibility_traces[profile] *= self.eligibility_decay
            if self.eligibility_traces[profile] < 0.01:
                del self.eligibility_traces[profile]

    def apply(self, performance: Dict[str, float], lifecycle: Dict[str, Dict], profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0) -> None:
        if profile not in performance:
            performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)
        modulated_update = self.lr * eligibility * reward * outcome * coactivation
        performance[profile] += modulated_update

        # Homeostatic scaling
        current_avg = sum(performance.values()) / max(len(performance), 1)
        if current_avg > self.homeostatic_target:
            performance[profile] *= 0.995

        if profile in lifecycle:
            lifecycle[profile]["performance_score"] = performance[profile]

        # Decay trace after use
        if profile in self.eligibility_traces:
            self.eligibility_traces[profile] *= 0.5
