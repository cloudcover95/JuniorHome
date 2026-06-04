# path: src/juniorhome/sovereign_edge_orchestrator.py
#!/usr/bin/env python3
"""
Sovereign Edge Orchestrator (Iteration 100 Capstone)

Final unifying layer of the entire JuniorCloud LLC architecture.

Provides a single, clean entry point that combines:
- AgentOrchestrator (with TriStateRouter + ManifoldFoldingQuantizer)
- JuniorAGI Container Agent Manager
- EdgeRuntime (efficiency + memory)
- SecondBrainPipeline (Event Sourcing + CQRS + knowledge)
- HardwareAbstraction (MLX / future backends)

This is the top-level interface for the sovereign edge ecosystem.
"""

import logging
from typing import Any, Dict, Optional

try:
    from .agent_orchestrator import AgentOrchestrator
    HAS_AGENT_ORCH = True
except ImportError:
    HAS_AGENT_ORCH = False

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

try:
    from .second_brain_pipeline import SecondBrainPipeline
    HAS_SECOND_BRAIN = True
except ImportError:
    HAS_SECOND_BRAIN = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SovereignEdgeOrchestrator:
    """
    Top-level orchestrator for the complete sovereign edge architecture.
    """

    def __init__(self):
        self.agent_orchestrator = AgentOrchestrator() if HAS_AGENT_ORCH else None
        self.junior_agi = ContainerAgentManager() if HAS_JUNIORAGI else None
        self.edge_runtime = EdgeRuntime() if HAS_EDGE else None
        self.second_brain_pipeline = SecondBrainPipeline(
            second_brain=self.agent_orchestrator.second_brain if self.agent_orchestrator else None
        ) if HAS_SECOND_BRAIN else None

        logging.info("SovereignEdgeOrchestrator initialized (Iteration 100 Capstone)")

    def think_and_act(self, prompt: str, task_complexity: str = "medium") -> Dict[str, Any]:
        if not self.agent_orchestrator:
            return {"error": "AgentOrchestrator not available"}
        return self.agent_orchestrator.autonomous_think_and_act(prompt, task_complexity=task_complexity)

    def analyze_manifold(self, state: Any) -> Dict[str, Any]:
        if not self.agent_orchestrator:
            return {"error": "AgentOrchestrator not available"}
        return self.agent_orchestrator.analyze_manifold(state)

    def route_intelligence(self, data: Any, mode: str = "auto", agent_context: Any = None):
        if not self.agent_orchestrator:
            return {"error": "AgentOrchestrator not available"}
        return self.agent_orchestrator.route_intelligence(data, mode=mode, agent_context=agent_context)

    def run_quant_team(self, svd_manifold: Any) -> Dict[str, Any]:
        if not self.agent_orchestrator:
            return {"error": "AgentOrchestrator not available"}
        return self.agent_orchestrator.run_quant_agent_team(svd_manifold)

    def spawn_container_agent(self, task_type: str, **kwargs) -> str:
        if not self.junior_agi:
            return "JuniorAGI not available"
        return self.junior_agi.spawn_agent(task_type, **kwargs)

    def send_command_to_agent(self, agent_id: str, command: str) -> Dict[str, Any]:
        if not self.junior_agi:
            return {"error": "JuniorAGI not available"}
        return self.junior_agi.send_command(agent_id, command)

    def get_status(self) -> Dict[str, Any]:
        status = {}
        if self.agent_orchestrator:
            status["agent_orchestrator"] = self.agent_orchestrator.get_full_status()
        if self.edge_runtime:
            status["edge"] = self.edge_runtime.get_status()
        if self.junior_agi:
            status["junior_agi_active_agents"] = len(self.junior_agi.active_agents)
        return status

    def shutdown(self):
        logging.info("SovereignEdgeOrchestrator shutting down...")
        if self.junior_agi:
            for agent_id in list(self.junior_agi.active_agents.keys()):
                self.junior_agi.stop_agent(agent_id)


# Convenience global instance for simple usage
sovereign_orchestrator = SovereignEdgeOrchestrator()
