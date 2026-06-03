# path: src/juniorhome/agent_orchestrator.py
#!/usr/bin/env python3
"""
Agent Orchestrator

Unified high-level orchestrator for autonomous agents.
Combines reasoning (JuniorLLM), efficiency (EdgeRuntime), knowledge (SecondBrainPipeline),
and execution into one clean interface.

This is the top-level coordination layer for deep agent autonomy.
"""

import logging
from typing import Any, Dict, Optional

try:
    from juniorllm.autonomy.autonomous_coder import AutonomousCoder
    HAS_JUNIORLLM = True
except ImportError:
    HAS_JUNIORLLM = False

from .autonomous_agent import AutonomousAgent
from .second_brain_pipeline import SecondBrainPipeline
from .edge_runtime import EdgeRuntime
from .second_brain import SecondBrain

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AgentOrchestrator:
    """
    Top-level orchestrator for autonomous agents with full stack integration.
    """

    def __init__(self, agent: Optional[AutonomousAgent] = None):
        self.agent = agent or AutonomousAgent()
        self.second_brain = SecondBrain(vault_path="./obsidian")
        self.pipeline = SecondBrainPipeline(second_brain=self.second_brain)
        self.edge_runtime = EdgeRuntime()

        if HAS_JUNIORLLM:
            from juniorllm.autonomy.autonomous_coder import AutonomousCoder
            self.coder = AutonomousCoder()
        else:
            self.coder = None

        logging.info("AgentOrchestrator initialized (unified deep architecture)")

    def autonomous_think_and_act(self, prompt: str, task_complexity: str = "medium") -> Dict[str, Any]:
        # Step 1: Think efficiently
        thought = self.agent.think(prompt, task_complexity=task_complexity)

        # Step 2: If it involves code, generate + review with efficiency in mind
        code_proposal = None
        if "code" in prompt.lower() or "implement" in prompt.lower():
            if self.coder:
                proposal = self.coder.propose_code_change(prompt)
                # Make generated code edge-efficient by default
                code_proposal = {
                    **proposal,
                    "edge_optimized": True,
                    "recommendations": self.edge_runtime.tinyml.get_optimization_tips(),
                }

        # Step 3: Store important thoughts in Second Brain
        if "important" in thought.lower() or len(thought) > 200:
            self.pipeline.second_brain.store_finding({
                "type": "agent_thought",
                "content": thought,
                "prompt": prompt,
            })

        return {
            "thought": thought,
            "code_proposal": code_proposal,
            "stored_in_second_brain": True,
        }

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
