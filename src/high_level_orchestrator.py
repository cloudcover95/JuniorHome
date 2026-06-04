# path: src/juniorhome/high_level_orchestrator.py
#!/usr/bin/env python3
"""
HighLevelOrchestrator (End-User Focused)

Provides simple, natural-language friendly interfaces so end users
never need to write code. Maps to TriStateExecutionEngine + kernel.
"""

import logging
from typing import Any, Dict, Optional

try:
    from .tri_state_execution_engine import TriStateExecutionEngine
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

try:
    from .sovereign_edge_orchestrator import SovereignEdgeOrchestrator
    HAS_SOVEREIGN = True
except ImportError:
    HAS_SOVEREIGN = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class HighLevelOrchestrator:
    def __init__(self):
        self.engine = TriStateExecutionEngine(
            agent_orchestrator=None,  # Will be resolved internally
            second_brain_pipeline=None
        ) if HAS_ENGINE else None

        self.sovereign = SovereignEdgeOrchestrator() if HAS_SOVEREIGN else None
        logging.info("HighLevelOrchestrator initialized (end-user friendly)")

    def _detect_intent(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["sovereignty", "private", "local", "secure"]):
            return "user"
        if any(word in prompt_lower for word in ["agent", "debate", "team", "discuss", "learn"]):
            return "swarm"
        if any(word in prompt_lower for word in ["verify", "check", "fallback", "safe"]):
            return "industry"
        return "auto"

    def run(self, prompt_or_task: str, data: Any = None) -> Dict[str, Any]:
        """
        Main end-user entry point. Supports natural language or task names.
        """
        if not self.engine:
            return {"error": "Execution engine not available"}

        mode = self._detect_intent(prompt_or_task)

        # Simple task name mapping
        task_lower = prompt_or_task.lower()
        if "fold" in task_lower or "manifold" in task_lower:
            mode = "user"
        elif "debate" in task_lower or "swarm" in task_lower:
            mode = "swarm"
        elif "kernel" in task_lower or "inject" in task_lower:
            mode = "user"  # Will inject after

        result = self.engine.execute(data or prompt_or_task, mode=mode)
        return result

    def run_task(self, task_type: str, data: Any = None) -> Dict[str, Any]:
        """
        Structured task interface (for UIs or simple scripts).
        """
        if not self.engine:
            return {"error": "Execution engine not available"}

        mode_map = {
            "user_analyze": "user",
            "user_fold": "user",
            "swarm_debate": "swarm",
            "swarm_learn": "swarm",
            "industry_verify": "industry",
            "kernel_inject": "user",
            "diagnostic": "auto",
        }
        mode = mode_map.get(task_type, "auto")
        result = self.engine.execute(data, mode=mode)
        return result

    def smart_execute(self, state: Any, goal: str = "balanced") -> Dict[str, Any]:
        """
        Let the system decide with goal awareness.
        """
        if not self.engine:
            return {"error": "Execution engine not available"}
        return self.engine.execute(state, mode="auto")

    def get_status(self) -> Dict[str, Any]:
        if self.sovereign:
            return self.sovereign.get_status()
        return {"status": "HighLevelOrchestrator active"}
