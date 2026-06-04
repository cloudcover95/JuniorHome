# path: src/juniorhome/agent_orchestrator.py
#!/usr/bin/env python3
"""
Agent Orchestrator (with JuniorQuant Integration)

Now includes the universal quant agent team from JuniorQuant.
Fully cross-pollinated with EdgeRuntime for efficient execution.
"""

import logging
from typing import Any, Dict, Optional

try:
    from juniorllm.autonomy.autonomous_coder import AutonomousCoder
    HAS_JUNIORLLM = True
except ImportError:
    HAS_JUNIORLLM = False

# JuniorQuant integration (universal quant black box)
try:
    from juniorquant.orchestration.agent_team import orchestrate_quant_team
    HAS_JUNIORQUANT = True
except ImportError:
    HAS_JUNIORQUANT = False
    orchestrate_quant_team = None

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

        if HAS_JUNIORLLM:
            from juniorllm.autonomy.autonomous_coder import AutonomousCoder
            self.coder = AutonomousCoder()
        else:
            self.coder = None

        logging.info("AgentOrchestrator initialized with JuniorQuant integration")

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
        """
        Runs the universal JuniorQuant agent team (Alpha → Risk → Execution).
        Executes efficiently via EdgeRuntime.
        """
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

        # Store result in Second Brain
        self.pipeline.second_brain.store_finding({
            "type": "quant_agent_result",
            "result": result,
        })

        return result

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
