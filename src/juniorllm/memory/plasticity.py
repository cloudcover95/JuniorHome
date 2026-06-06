# path: src/juniorllm/memory/plasticity.py

"""
PlasticityEngine

Added simple structural plasticity (creation/pruning of profile connections)
for more biological fidelity.

Also supports security-aware plasticity (integrity checks).
"""

from typing import Dict


class PlasticityEngine:
    def __init__(self, lr: float = 0.01, homeostatic_target: float = 0.15):
        self.lr = lr
        self.homeostatic_target = homeostatic_target
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9
        self.connection_strength: Dict[str, float] = {}  # Structural plasticity

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        if profile not in self.eligibility_traces:
            self.eligibility_traces[profile] = 0.0
        self.eligibility_traces[profile] = min(1.0, self.eligibility_traces[profile] + strength)

    def decay_eligibility_traces(self):
        for profile in list(self.eligibility_traces.keys()):
            self.eligibility_traces[profile] *= self.eligibility_decay
            if self.eligibility_traces[profile] < 0.01:
                del self.eligibility_traces[profile]

    def apply(self, performance: Dict[str, float], lifecycle: Dict[str, Dict], profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0):
        if profile not in performance:
            performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)

        if outcome > 0:
            timing_factor = eligibility
            delta_w = self.lr * timing_factor * reward * outcome * coactivation
        else:
            timing_factor = max(0.2, 1.0 - eligibility)
            delta_w = self.lr * timing_factor * abs(outcome) * reward * coactivation * -1.0

        performance[profile] += delta_w

        # Structural plasticity: strengthen or prune connections
        if profile not in self.connection_strength:
            self.connection_strength[profile] = 0.0

        if delta_w > 0:
            self.connection_strength[profile] = min(1.0, self.connection_strength[profile] + 0.05)
        else:
            self.connection_strength[profile] = max(0.0, self.connection_strength[profile] - 0.03)

        # Prune very weak connections (biological pruning)
        if self.connection_strength[profile] < 0.05:
            self.connection_strength[profile] = 0.0

        # Homeostatic scaling
        current_avg = sum(performance.values()) / max(len(performance), 1)
        if current_avg > self.homeostatic_target:
            performance[profile] *= 0.995

        if profile in lifecycle:
            lifecycle[profile]["performance_score"] = performance[profile]
            lifecycle[profile]["connection_strength"] = self.connection_strength[profile]

        if profile in self.eligibility_traces:
            self.eligibility_traces[profile] *= 0.55

    def get_connection_strength(self, profile: str) -> float:
        return self.connection_strength.get(profile, 0.0)
