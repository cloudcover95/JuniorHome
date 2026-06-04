# path: src/juniorhome/agent_orchestrator.py
#!/usr/bin/env python3
"""
Agent Orchestrator (TriStateRouter Integrated)

Now uses the hardware-agnostic TriStateRouter for smart routing decisions.
Routing is efficiency-aware via EdgeRuntime.
"""

import logging
from typing import Any, Dict, Optional

try:
    from juniorllm.autonomy.autonomous_coder import AutonomousCoder
    HAS_JUNIORLLM = True
except ImportError:
    HAS_JUNIORLLM = False

try:
    from juniorquant.orchestration.agent_team import orchestrate_quant_team
    HAS_JUNIORQUANT = True
except ImportError:
    HAS_JUNIORQUANT = False
    orchestrate_quant_team = None

try:
    from bitnet_mlx.compute.tri_state_router import TriStateRouter
    HAS_TRISTATE = True
except ImportError:
    HAS_TRISTATE = False
    TriStateRouter = None


from .autonomous_agent import AutonomousAgent
from .second_brain_pipeline import SecondBrainPipeline
from .edge_runtime import EdgeRuntime
from .second_brain import SecondBrain
from .security_middleware import SecurityMiddleware

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AgentOrchestrator:
    def __init__(self, agent: Optional[AutonomousAgent] = None):
        self.agent = agent or AutonomousAgent()
        self.second_brain = SecondBrain(vault_path="./obsidian")
        self.pipeline = SecondBrainPipeline(second_brain=self.second_brain)
        self.edge_runtime = EdgeRuntime()
        self.security = SecurityMiddleware()

        if HAS_TRISTATE:
            self.router = TriStateRouter()
        else:
            self.router = None

        if HAS_JUNIORLLM:
            from juniorllm.autonomy.autonomous_coder import AutonomousCoder
            self.coder = AutonomousCoder()
        else:
            self.coder = None

        logging.info("AgentOrchestrator initialized with TriStateRouter")

    def autonomous_think_and_act(self, prompt: str, task_complexity: str = "medium") -> Dict[str, Any]:
        thought = self.agent.think(prompt, task_complexity=task_complexity)

        code_proposal = None
        if "code" in prompt.lower() or "implement" in prompt.lower():
            if self.coder:
                raw_proposal = self.coder.propose_code_change(prompt)
                secured = self.security.secure_action(
                    lambda: raw_proposal,
                    llm_output=str(raw_proposal),
                    action_description="Code generation",
                )
                code_proposal = {
                    **raw_proposal,
                    "security_validated": secured.get("executed", False),
                    "edge_optimized": True,
                }

        if len(thought) > 150:
            self.pipeline.second_brain.store_finding({
                "type": "agent_thought",
                "content": thought,
                "prompt": prompt,
            })

        return {
            "thought": thought,
            "code_proposal": code_proposal,
            "stored_in_second_brain": len(thought) > 150,
        }

    def run_quant_agent_team(self, svd_manifold) -> Dict[str, Any]:
        if not HAS_JUNIORQUANT or orchestrate_quant_team is None:
            return {"error": "JuniorQuant not available"}

        def _execute_team():
            return orchestrate_quant_team(svd_manifold)

        result = self.edge_runtime.execute_efficiently(
            _execute_team,
            task_name="quant_agent_team",
            estimated_memory_mb=80,
            task_complexity="high",
        )

        self.pipeline.second_brain.store_finding({
            "type": "quant_agent_result",
            "result": result,
        })

        return result

    def route_intelligence(self, data: Any, mode: str = "auto", agent_context: Any = None):
        """
        Uses TriStateRouter for hardware-agnostic intelligent routing.
        Respects edge efficiency.
        """
        if not self.router:
            return {"error": "TriStateRouter not available"}

        def _route():
            return self.router.evaluate_and_route(data, agent_context=agent_context, mode=mode)

        return self.edge_runtime.execute_efficiently(
            _route,
            task_name="tri_state_routing",
            estimated_memory_mb=40,
            task_complexity="medium",
        )

    def process_knowledge(self, url: str = None):
        if url:
            return self.pipeline.process_rss_feed(url)
        return self.pipeline.get_stats()

    def get_full_status(self):
        return {
            "agent": self.agent.status(),
            "pipeline_stats": self.pipeline.get_stats(),
            "edge_status": self.edge_runtime.get_status(),
        }
