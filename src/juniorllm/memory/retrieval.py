# path: src/juniorllm/memory/retrieval.py

"""
MemoryRetriever

Composable component for context-dependent memory retrieval.
Can be extended with different scoring strategies.
"""

import time
from typing import Any, Dict, List, Optional


class MemoryRetriever:
    def __init__(self):
        pass

    def retrieve(self, history: List[Dict[str, Any]], current_profile: Optional[str] = None, context: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not history:
            return []

        scored = []
        for record in history:
            score = record.get("performance_at_activation", 0)

            if current_profile and record.get("active_profile") == current_profile:
                score += 0.15

            if context:
                if context.get("level") and record.get("level") == context.get("level"):
                    score += 0.1

                if context.get("prefer_recent"):
                    age = time.time() - record.get("timestamp", 0)
                    if age < 3600:
                        score += 0.08

            scored.append((score, record))

        scored.sort(reverse=True)
        return [r for score, r in scored[:top_k]]
