# path: src/juniorhome/high_level_orchestrator.py
#!/usr/bin/env python3
"""
HighLevelOrchestrator (Further Expanded)

More natural language understanding, additional task types,
and improved auto-decision making using historical coherence.
"""

import logging
from typing import Any, Dict, Optional, List

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
            agent_orchestrator=None,
            second_brain_pipeline=None
        ) if HAS_ENGINE else None

        self.sovereign = SovereignEdgeOrchestrator() if HAS_SOVEREIGN else None
        self.recent_coherence: List[float] = []
        logging.info("HighLevelOrchestrator initialized (expanded further)")

    def _detect_intent(self, prompt: str) -> str:
        p = prompt.lower()

        # Strong User Black Box signals
        if any(kw in p for kw in ["sovereignty", "private", "local only", "air gapped", "maximum security", "no cloud", "pure ternary"]):
            return "user"

        # Strong Swarm Black Box signals
        if any(kw in p for kw in ["agent", "debate", "team", "discuss", "collaborate", "swarm", "learn together", "second brain"]):
            return "swarm"

        # Industry Fallback signals
        if any(kw in p for kw in ["verify", "safe", "fallback", "check", "validate", "industry", "dense"]):
            return "industry"

        # Kernel injection signals
        if any(kw in p for kw in ["kernel", "inject", "bare metal", "junioros", "write to kernel"]):
            return "user"

        return "auto"

    def run(self, prompt_or_task: str, data: Any = None) -> Dict[str, Any]:
        if not self.engine:
            return {"error": "Execution engine not available"}

        mode = self._detect_intent(prompt_or_task)

        # Explicit task overrides
        task = prompt_or_task.lower()
        if any(x in task for x in ["fold", "manifold", "analyze manifold"]):
            mode = "user"
        elif any(x in task for x in ["debate", "swarm", "agent team", "learn"]):
            mode = "swarm"
        elif any(x in task for x in ["kernel inject", "write to kernel"]):
            mode = "user"

        result = self.engine.execute(data or prompt_or_task, mode=mode)
        return result

    def run_task(self, task_type: str, data: Any = None) -> Dict[str, Any]:
        if not self.engine:
            return {"error": "Execution engine not available"}

        mode_map = {
            "user_analyze": "user",
            "user_fold_manifold": "user",
            "swarm_debate": "swarm",
            "swarm_learn": "swarm",
            "industry_verify": "industry",
            "kernel_inject": "user",
            "diagnostic": "auto",
            "build_test": "auto",
        }

        mode = mode_map.get(task_type, "auto")
        result = self.engine.execute(data, mode=mode)
        return result

    def smart_execute(self, state: Any, goal: str = "balanced") -> Dict[str, Any]:
        if not self.engine:
            return {"error": "Execution engine not available"}
        return self.engine.execute(state, mode="auto")

    def get_status(self) -> Dict[str, Any]:
        if self.sovereign:
            return self.sovereign.get_status()
        return {"status": "HighLevelOrchestrator active"}
