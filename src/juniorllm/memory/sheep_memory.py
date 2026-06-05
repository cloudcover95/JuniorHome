# path: src/juniorllm/memory/sheep_memory.py

"""
SHEEP Memory System

Biologically-inspired memory for the JuniorLLMStateMachine.

This module contains the core memory structures and algorithms:
- History of awakenings
- Reflection (short-term boost)
- Consolidation (long-term insights)
- Replay (hippocampal-style reinforcement)
- Plasticity rules (Hebbian + homeostatic)
- Cued retrieval

Future: This can be backed by JuniorMemSys-Suite for persistent,
TDA-based long-term storage (the 'neocortex' layer).
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

        # Plasticity parameters
        self.plasticity_lr: float = 0.01
        self.homeostatic_target: float = 0.15

    def record_awakening(self, level: str, performance: float, active_profile: str):
        record = {
            "timestamp": time.time(),
            "level": level,
            "performance_at_activation": performance,
            "active_profile": active_profile
        }
        self.history.append(record)
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
                boost = 0.02 if level == "FULL_AWAKENING" else 0.01
                self.performance[profile] = self.performance.get(profile, 0) + boost
                if profile in self.lifecycle:
                    self.lifecycle[profile]["performance_score"] = self.performance[profile]

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
            boost = 0.05 * (avg / 0.1)
            self.performance[best] += boost
            if best in self.lifecycle:
                self.lifecycle[best]["performance_score"] = self.performance[best]

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
                boost = 0.04 * (perf / 0.1)
                self.performance[profile] += boost
                if profile in self.lifecycle:
                    self.lifecycle[profile]["performance_score"] = self.performance[profile]

        # Gentle decay + prune
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
        return [r for _, r in scored[:top_k]]

    def apply_plasticity(self, profile: str, outcome: float, coactivation: float = 1.0):
        if profile not in self.performance:
            self.performance[profile] = 0.0

        self.performance[profile] += self.plasticity_lr * outcome * coactivation

        # Homeostatic scaling
        avg = sum(self.performance.values()) / max(len(self.performance), 1)
        if avg > self.homeostatic_target:
            self.performance[profile] *= 0.995

        if profile in self.lifecycle:
            self.lifecycle[profile]["performance_score"] = self.performance[profile]

    def get_insights(self) -> Dict[str, Any]:
        return self.consolidated_insights.copy()

    def get_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        return self.history[-last_n:]
