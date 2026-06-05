# path: src/juniorllm/memory/sheep_memory.py

"""
SHEEPMemory

Advanced biologically-inspired memory system with:
- Multi-scale consolidation
- Eligibility traces + reward-modulated plasticity
- Sleep-like offline consolidation
- Improved context-dependent retrieval

Designed for future integration with JuniorMemSys-Suite.
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

        self.consolidation_scales: Dict[int, Dict[str, Any]] = {
            0: {"last_run": 0, "interval": 1},
            1: {"last_run": 0, "interval": 5},
            2: {"last_run": 0, "interval": 20},
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
        if scale == 0:
            self.reflect_on_recent()
            return
        if scale == 1:
            self._systems_consolidation()
        elif scale == 2:
            self._meta_consolidation()

    def _systems_consolidation(self):
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
        if len(self.history) < 10:
            return

        high_level_count = len([r for r in self.history if r.get("level") in ("ELEVATED", "FULL_AWAKENING")])
        avg_perf = sum(r.get("performance_at_activation", 0) for r in self.history) / len(self.history)

        meta = {
            "total_awakenings": len(self.history),
            "high_level_ratio": high_level_count / len(self.history),
            "average_performance": round(avg_perf, 4),
            "last_meta_consolidation": time.time(),
            "scale": 2
        }

        self.consolidated_insights.update(meta)
        self.backend.store_consolidated_insights(self.consolidated_insights)

        print(f"[SHEEP Meta Consolidation] Global insights updated. High-level ratio: {meta['high_level_ratio']:.2f}")

    def sleep_like_offline_consolidation(self, iterations: int = 3):
        """Sleep-like offline consolidation.

        Biological inspiration: During 'sleep' (offline periods), the brain replays
        experiences to consolidate memories without new input.

        This method performs multiple rounds of replay + deeper consolidation
        on existing high-value memories. It improves long-term retention and
        generalization with no external data required.
        """
        if len(self.history) < 5:
            return

        print(f"[SHEEP Sleep] Starting offline consolidation ({iterations} iterations)...")

        for i in range(iterations):
            # Replay strongest memories with high reward
            high_level = [r for r in self.history if r.get("level") in ("ELEVATED", "FULL_AWAKENING")]
            if high_level:
                # Sort by performance and take top ones
                sorted_memories = sorted(high_level, key=lambda r: r.get("performance_at_activation", 0), reverse=True)
                for mem in sorted_memories[:3]:  # Replay top 3
                    profile = mem.get("active_profile")
                    perf = mem.get("performance_at_activation", 0)
                    if profile and profile in self.performance:
                        # Strong offline reinforcement
                        self.apply_plasticity(profile, outcome=perf, reward=2.5)

            # Run deeper systems + meta consolidation
            self._systems_consolidation()
            if i == iterations - 1:
                self._meta_consolidation()

            # Decay traces (like memory fading during sleep)
            self.decay_eligibility_traces()

        print("[SHEEP Sleep] Offline consolidation complete.")

    def retrieve_relevant(self, current_profile: Optional[str] = None, context: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Improved context-dependent retrieval.

        Now supports optional context dict for smarter scoring
        (e.g., recent state, current performance level, or external cues).
        """
        if not self.history:
            return []

        scored = []
        for r in self.history:
            score = r.get("performance_at_activation", 0)

            # Base boost if same profile is currently active
            if current_profile and r.get("active_profile") == current_profile:
                score += 0.15

            # Context-aware boosts
            if context:
                # Boost if awakening happened in similar 'level' context
                if context.get("level") and r.get("level") == context.get("level"):
                    score += 0.1

                # Boost recent memories if 'recency' context is high
                if context.get("prefer_recent"):
                    age = time.time() - r.get("timestamp", 0)
                    if age < 3600:  # Within last hour
                        score += 0.08

            scored.append((score, r))

        scored.sort(reverse=True)
        return [r for score, r in scored[:top_k]]

    def get_insights(self) -> Dict[str, Any]:
        return self.consolidated_insights.copy()

    def get_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        return self.history[-last_n:]
