# path: src/juniorhome/sovereign_edge_orchestrator.py
#!/usr/bin/env python3
"""
SovereignEdgeOrchestrator (with TriStateExecutionEngine)

Now uses the dedicated TriStateExecutionEngine for clean separation
of the three black boxes defined in the JUNIOR_OS_DIRECTIVE.

Includes hooks for future JuniorOS kernel injection (/dev/junior_spark).
"""

import logging
from typing import Any, Dict, Optional

try:
    from .tri_state_execution_engine import TriStateExecutionEngine
    HAS_TRISTATE_ENGINE = True
except ImportError:
    HAS_TRISTATE_ENGINE = False

try:
    from .junioragi.container_agent_manager import ContainerAgentManager
    HAS_JUNIORAGI = True
except ImportError:
    HAS_JUNIORAGI = False

try:
    from .edge_runtime import EdgeRuntime
    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SovereignEdgeOrchestrator:
    def __init__(self):
        self.tri_state_engine = TriStateExecutionEngine(
            agent_orchestrator=self._get_agent_orchestrator()
        ) if HAS_TRISTATE_ENGINE else None

        self.junior_agi = ContainerAgentManager() if HAS_JUNIORAGI else None
        self.edge_runtime = EdgeRuntime() if HAS_EDGE else None

        logging.info("SovereignEdgeOrchestrator initialized with TriStateExecutionEngine")

    def _get_agent_orchestrator(self):
        try:
            from .agent_orchestrator import AgentOrchestrator
            return AgentOrchestrator()
        except ImportError:
            return None

    def execute(self, state: Any, mode: str = "auto", agent_context: Any = None) -> Dict[str, Any]:
        """
        Main entry point. Routes to the appropriate black box.
        """
        if not self.tri_state_engine:
            return {"error": "TriStateExecutionEngine not available"}

        result = self.tri_state_engine.execute(state, mode=mode, agent_context=agent_context)

        # Optional: inject result into kernel when JuniorOS is available
        self._try_kernel_injection(result)

        return result

    def _try_kernel_injection(self, result: Dict[str, Any]):
        """
        Placeholder for future /dev/junior_spark kernel injection.
        In production this would write ternary manifolds to the kernel ring buffer.
        """
        if result.get("status") == "KERNEL_INJECTED":
            logging.debug("Result already injected to kernel")
        # Future: integrate with KernelBridge

    def analyze_manifold(self, state: Any) -> Dict[str, Any]:
        if not self.tri_state_engine:
            return {"error": "TriStateExecutionEngine not available"}
        return self.tri_state_engine.user_box.execute(state)  # Use User box for pure analysis

    def spawn_container_agent(self, task_type: str, **kwargs) -> str:
        if not self.junior_agi:
            return "JuniorAGI not available"
        return self.junior_agi.spawn_agent(task_type, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        status = {}
        if self.tri_state_engine:
            status["tri_state_boxes"] = self.tri_state_engine.get_box_status()
        if self.edge_runtime:
            status["edge"] = self.edge_runtime.get_status()
        if self.junior_agi:
            status["active_container_agents"] = len(self.junior_agi.active_agents)
        return status

    def shutdown(self):
        logging.info("SovereignEdgeOrchestrator shutting down...")
        if self.junior_agi:
            for agent_id in list(self.junior_agi.active_agents.keys()):
                self.junior_agi.stop_agent(agent_id)
