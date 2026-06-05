# path: src/juniorllm/memory/sheep_memory.py

"""
SHEEP Memory System

Biologically-inspired memory + learning for JuniorLLM.

Core features:
- History, Reflection, Consolidation, Replay
- Plasticity rules with eligibility traces and reward modulation
- Cued retrieval

Designed to eventually integrate with JuniorMemSys-Suite for persistent storage.
"""

import time
import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SHEEPLevel:
    INACTIVE = 0
    BASIC = 1
    ELEVATED = 2
    FULL_AWAKENING = 3


class SHEEPMemory:
    def __init__(self, node_id: str = "default"):
        self.node_id = node_id
        self.history: List[Dict[str, Any]] = []
        self.consolidated_insights: Dict[str, Any] = {}
        self.performance: Dict[str, float] = {}
        self.lifecycle: Dict[str, Dict[str, Any]] = {}

        # Eligibility traces (decaying memory of recent activity)
        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9          # Decay factor per step

        # Plasticity parameters
        self.plasticity_lr: float = 0.01
        self.homeostatic_target: float = 0.15

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        """Update eligibility trace for a profile (like pre-synaptic activity)."""
        if profile not in self.eligibility_traces:
            self.eligibility_traces[profile] = 0.0
        self.eligibility_traces[profile] = min(1.0, self.eligibility_traces[profile] + strength)

    def decay_eligibility_traces(self):
        """Decay all eligibility traces (biological time constant)."""
        for profile in list(self.eligibility_traces.keys()):
            self.eligibility_traces[profile] *= self.eligibility_decay
            if self.eligibility_traces[profile] < 0.01:
                del self.eligibility_traces[profile]

    def apply_plasticity(self, profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0):
        """Deepened biologically-inspired plasticity rule.

        Combines:
        - Eligibility trace (credit assignment over time)
        - Reward modulation (stronger plasticity on positive outcomes)
        - Hebbian co-activation
        - Homeostatic scaling

        This approximates reward-modulated STDP / three-factor plasticity rules.
        """
        if profile not in self.performance:
            self.performance[profile] = 0.0

        # Get current eligibility trace (decayed memory of recent activity)
        eligibility = self.eligibility_traces.get(profile, 0.0)

        # Modulated update: eligibility * reward * outcome
        modulated_update = self.plasticity_lr * eligibility * reward * outcome * coactivation
        self.performance[profile] += modulated_update

        # Homeostatic scaling
        current_avg = sum(self.performance.values()) / max(len(self.performance), 1)
        if current_avg > self.homeostatic_target:
            self.performance[profile] *= 0.995

        if profile in self.lifecycle:
            self.lifecycle[profile]["performance_score"] = self.performance[profile]

        # Decay trace after use (biological reset)
        if profile in self.eligibility_traces:
            self.eligibility_traces[profile] *= 0.5

    def record_awakening(self, level: str, performance: float, active_profile: str):
        record = {
            "timestamp": time.time(),
            "level": level,
            "performance_at_activation": performance,
            "active_profile": active_profile
        }
        self.history.append(record)

        # Update eligibility trace when a profile is active during an event
        self.update_eligibility_trace(active_profile, strength=1.0)
        return record

    def reflect_on_recent(self):
        if not self.history:
            return
        latest = self.history[-1]
        level = latest.get("level")
        perf = latest.get("performance_at_activation", 0)

        if level in ("ELEVATED", "FULL_AWAKENING") and perf > 0.05:
            profile = latest.get("active_profile")
            if profile:
                # Use reward-modulated plasticity for reflection
                self.apply_plasticity(profile, outcome=perf, reward=1.2 if level == "FULL_AWAKENING" else 1.0)

    def consolidate(self):
        if len(self.history) < 3:
            return

        high_level = [r for r in self.history if r.get("level") in ("ELEVATED", "FULL_AWAKENING")]
        if not high_level:
            return

        scores = {}
        for r in high_level:
            p = r.get("active_profile")
            if p:
                scores[p] = scores.get(p, 0) + r.get("performance_at_activation", 0)

        if not scores:
            return

        best = max(scores, key=scores.get)
        avg = scores[best] / len(high_level)

        if best in self.performance:
            # Stronger modulated update during consolidation
            self.apply_plasticity(best, outcome=avg, reward=1.5)

            self.consolidated_insights = {
                "best_profile_over_time": best,
                "average_high_level_performance": round(avg, 4),
                "last_consolidation": time.time(),
                "total_awakenings_analyzed": len(high_level)
            }

    def replay_and_consolidate(self):
        if len(self.history) < 5:
            return

        high_level = [r for r in self.history if r.get("level") in ("ELEVATED", "FULL_AWAKENING")]
        if high_level:
            best = max(high_level, key=lambda r: r.get("performance_at_activation", 0))
            profile = best.get("active_profile")
            perf = best.get("performance_at_activation", 0)

            if profile and profile in self.performance and perf > 0.08:
                # Replay uses strong reward modulation
                self.apply_plasticity(profile, outcome=perf, reward=2.0)

        # Decay + prune
        self.decay_eligibility_traces()

        if len(self.history) > 25:
            for old in self.history[:-25]:
                if old.get("performance_at_activation", 0) < 0.08:
                    p = old.get("active_profile")
                    if p and p in self.performance:
                        self.performance[p] *= 0.98
            self.history = self.history[-20:]

    def retrieve_relevant(self, current_profile: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.history:
            return []
        scored = []
        for r in self.history:
            score = r.get("performance_at_activation", 0)
            if current_profile and r.get("active_profile") == current_profile:
                score += 0.1
            scored.append((score, r))
        scored.sort(reverse=True)
        return [r for score, r in scored[:top_k]]

    def get_insights(self) -> Dict[str, Any]:
        return self.consolidated_insights.copy()

    def get_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        return self.history[-last_n:]
