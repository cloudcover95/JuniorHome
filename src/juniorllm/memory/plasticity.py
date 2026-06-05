# path: src/juniorllm/memory/plasticity.py

"""
PlasticityEngine

Biologically-inspired plasticity with:
- Eligibility traces (for timing/credit assignment)
- Reward modulation
- STDP-style timing (potentiation when pre-synaptic activity is recent)
- Homeostatic scaling
- Support for depression on negative outcomes

This approximates reward-modulated STDP / three-factor learning rules.
"""

from typing import Dict


class PlasticityEngine:
    def __init__(self, lr: float = 0.01, homeostatic_target: float = 0.15):
        self.lr = lr
        self.homeostatic_target = homeostatic_target
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        """Record recent activity of a profile (pre-synaptic like)."""
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
        Apply plasticity with STDP-style timing.

        - The current eligibility trace value acts as a timing signal:
          High trace = recent pre-synaptic activity → strong potentiation (STDP-like LTP)
          Low/zero trace = activity was long ago → weak effect

        - Reward modulates overall strength.
        - Negative outcome can cause depression (LTD).
        - Homeostatic scaling prevents runaway growth.
        """
        if profile not in performance:
            performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)

        # STDP-style: timing-dependent potentiation
        # Stronger update when eligibility is high (recent activity before outcome)
        timing_factor = eligibility  # 0.0 to 1.0

        # Allow depression for negative outcomes
        effective_outcome = outcome
        if outcome < 0:
            # Long-term depression (weaker when trace is high, classic STDP)
            timing_factor = max(0.1, 1.0 - eligibility)  # inverse timing for depression

        modulated_update = self.lr * timing_factor * reward * effective_outcome * coactivation
        performance[profile] += modulated_update

        # Homeostatic scaling
        current_avg = sum(performance.values()) / max(len(performance), 1)
        if current_avg > self.homeostatic_target:
            performance[profile] *= 0.995

        if profile in lifecycle:
            lifecycle[profile]["performance_score"] = performance[profile]

        # Decay trace after plasticity application (biological reset)
        if profile in self.eligibility_traces:
            self.eligibility_traces[profile] *= 0.6
