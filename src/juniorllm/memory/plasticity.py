# path: src/juniorllm/memory/plasticity.py

"""
PlasticityEngine

Investigation into STDP (Spike-Timing-Dependent Plasticity) algorithms.

Current implementation approximates reward-modulated STDP using:
- Eligibility traces as timing signal (proxy for pre-synaptic activity recency)
- Exponential-like dependence via trace strength
- Potentiation when pre before post (high eligibility + positive outcome)
- Depression when timing is reversed or outcome is negative

This is a lightweight, abstract version suitable for sovereign edge systems.
"""

from typing import Dict


class PlasticityEngine:
    def __init__(self, lr: float = 0.01, homeostatic_target: float = 0.15):
        self.lr = lr
        self.homeostatic_target = homeostatic_target
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        """Record recent 'pre-synaptic' activity of a profile."""
        if profile not in self.eligibility_traces:
            self.eligibility_traces[profile] = 0.0
        self.eligibility_traces[profile] = min(1.0, self.eligibility_traces[profile] + strength)

    def decay_eligibility_traces(self):
        for profile in list(self.eligibility_traces.keys()):
            self.eligibility_traces[profile] *= self.eligibility_decay
            if self.eligibility_traces[profile] < 0.01:
                del self.eligibility_traces[profile]

    def apply(self, performance: Dict[str, float], lifecycle: Dict[str, Dict], profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0):
        """
        Apply STDP-inspired plasticity rule.

        Biological mapping:
        - Eligibility trace = decaying memory of recent pre-synaptic activity
        - Positive outcome + high eligibility = pre before post → LTP (potentiation)
        - Negative outcome or low eligibility = post before pre or weak timing → LTD (depression)

        The update strength depends on how recent the activity was (trace value).
        """
        if profile not in performance:
            performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)

        # STDP-style timing window
        if outcome > 0:
            # Potentiation window: stronger when eligibility is high (recent pre-activity)
            timing_factor = eligibility
            delta_w = self.lr * timing_factor * reward * outcome * coactivation
        else:
            # Depression window: stronger when eligibility is low (reversed or weak timing)
            timing_factor = max(0.2, 1.0 - eligibility)
            delta_w = self.lr * timing_factor * abs(outcome) * reward * coactivation * -1.0

        performance[profile] += delta_w

        # Homeostatic scaling (synaptic scaling)
        current_avg = sum(performance.values()) / max(len(performance), 1)
        if current_avg > self.homeostatic_target:
            performance[profile] *= 0.995

        if profile in lifecycle:
            lifecycle[profile]["performance_score"] = performance[profile]

        # Decay trace after plasticity (biological reset after learning event)
        if profile in self.eligibility_traces:
            self.eligibility_traces[profile] *= 0.55
