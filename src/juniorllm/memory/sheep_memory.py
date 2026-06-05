# path: src/juniorllm/memory/sheep_memory.py

"""
SHEEPMemory with Multi-Scale Consolidation

Biological inspiration:
- Fast synaptic consolidation (immediate reflection)
- Systems consolidation (replay + pattern extraction over multiple events)
- Long-term / meta consolidation (global strategy updates over many sessions)

This implements multi-scale memory consolidation.
"""

import time
import logging
from typing import Any, Dict, List, Optional

from .memory_backend import MemoryBackend, InMemoryBackend

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SHEEPLevel:
    INACTIVE = 0
    BASIC = 1
    ELEVATED = 2
    FULL_AWAKENING = 3


class SHEEPMemory:
    def __init__(self, node_id: str = "default", backend: Optional[MemoryBackend] = None):
        self.node_id = node_id
        self.backend = backend or InMemoryBackend()

        self.history: List[Dict[str, Any]] = []
        self.consolidated_insights: Dict[str, Any] = {}
        self.performance: Dict[str, float] = {}
        self.lifecycle: Dict[str, Dict[str, Any]] = {}

        self.eligibility_traces: Dict[str, float] = {}
        self.eligibility_decay: float = 0.9
        self.plasticity_lr: float = 0.01
        self.homeostatic_target: float = 0.15

        # Multi-scale state
        self.consolidation_scales: Dict[int, Dict[str, Any]] = {
            0: {"last_run": 0, "interval": 1},      # Fast / immediate
            1: {"last_run": 0, "interval": 5},      # Medium / systems
            2: {"last_run": 0, "interval": 20},     # Long-term / meta
        }

        self._sync_from_backend()

    def _sync_from_backend(self):
        self.history = self.backend.get_history(last_n=100)
        self.consolidated_insights = self.backend.get_consolidated_insights()

    def update_eligibility_trace(self, profile: str, strength: float = 1.0):
        if profile not in self.eligibility_traces:
            self.eligibility_traces[profile] = 0.0
        self.eligibility_traces[profile] = min(1.0, self.eligibility_traces[profile] + strength)

    def decay_eligibility_traces(self):
        for profile in list(self.eligibility_traces.keys()):
            self.eligibility_traces[profile] *= self.eligibility_decay
            if self.eligibility_traces[profile] < 0.01:
                del self.eligibility_traces[profile]

    def apply_plasticity(self, profile: str, outcome: float, reward: float = 1.0, coactivation: float = 1.0):
        if profile not in self.performance:
            self.performance[profile] = 0.0

        eligibility = self.eligibility_traces.get(profile, 0.0)
        modulated_update = self.plasticity_lr * eligibility * reward * outcome * coactivation
        self.performance[profile] += modulated_update

        current_avg = sum(self.performance.values()) / max(len(self.performance), 1)
        if current_avg > self.homeostatic_target:
            self.performance[profile] *= 0.995

        if profile in self.lifecycle:
            self.lifecycle[profile]["performance_score"] = self.performance[profile]

        self.backend.store_performance(profile, self.performance[profile])
        if profile in self.lifecycle:
            self.backend.store_lifecycle(profile, self.lifecycle[profile])

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
        self.backend.store_awakening(record)

        self.update_eligibility_trace(active_profile, strength=1.0)
        return record

    def reflect_on_recent(self):
        """Fast scale (Scale 0) - immediate synaptic-like consolidation."""
        if not self.history:
            return
        latest = self.history[-1]
        level = latest.get("level")
        perf = latest.get("performance_at_activation", 0)

        if level in ("ELEVATED", "FULL_AWAKENING") and perf > 0.05:
            profile = latest.get("active_profile")
            if profile:
                self.apply_plasticity(profile, outcome=perf, reward=1.2 if level == "FULL_AWAKENING" else 1.0)

    def consolidate(self, scale: int = 1):
        """Multi-scale consolidation.

        scale=0: Fast / immediate (handled in reflect_on_recent)
        scale=1: Medium / systems consolidation (pattern extraction from recent history)
        scale=2: Long-term / meta consolidation (global insights, strategy updates)
        """
        if scale == 0:
            self.reflect_on_recent()
            return

        if scale == 1:
            self._systems_consolidation()
        elif scale == 2:
            self._meta_consolidation()

    def _systems_consolidation(self):
        """Medium-scale systems consolidation (inspired by hippocampal-neocortical transfer)."""
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
            self.apply_plasticity(best, outcome=avg, reward=1.5)

            self.consolidated_insights = {
                "best_profile_over_time": best,
                "average_high_level_performance": round(avg, 4),
                "last_consolidation": time.time(),
                "total_awakenings_analyzed": len(high_level),
                "scale": 1
            }
            self.backend.store_consolidated_insights(self.consolidated_insights)

    def _meta_consolidation(self):
        """Long-term meta consolidation (global strategy / meta-learning level)."""
        if len(self.history) < 10:
            return

        # Analyze overall trends across all scales
        high_level_count = len([r for r in self.history if r.get("level") in ("ELEVATED", "FULL_AWAKENING")])
        avg_perf = sum(r.get("performance_at_activation", 0) for r in self.history) / len(self.history)

        # Update global meta-insights
        meta = {
            "total_awakenings": len(self.history),
            "high_level_ratio": high_level_count / len(self.history),
            "average_performance": round(avg_perf, 4),
            "last_meta_consolidation": time.time(),
            "scale": 2
        }

        # Merge with existing consolidated insights
        self.consolidated_insights.update(meta)
        self.backend.store_consolidated_insights(self.consolidated_insights)

        print(f"[SHEEP Meta Consolidation] Global insights updated. High-level ratio: {meta['high_level_ratio']:.2f}")

    def replay_and_consolidate(self):
        if len(self.history) < 5:
            return

        high_level = [r for r in self.history if r.get("level") in ("ELEVATED", "FULL_AWAKENING")]
        if high_level:
            best = max(high_level, key=lambda r: r.get("performance_at_activation", 0))
            profile = best.get("active_profile")
            perf = best.get("performance_at_activation", 0)

            if profile and profile in self.performance and perf > 0.08:
                self.apply_plasticity(profile, outcome=perf, reward=2.0)

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
