# path: src/juniorhome/autonomous_agent.py
#!/usr/bin/env python3
"""
Autonomous Agent

High-level agent abstraction for JuniorHome.
Combines LLM routing, task execution, memory access, and scheduling
into a single manageable object. Designed for building autonomous
workflows and agents on top of the sovereign stack.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .smart_llm_router import SmartLLMRouter
from .task_runner import TaskRunner
from .workflow_engine import WorkflowEngine

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AutonomousAgent:
    """
    High-level autonomous agent.
    """

    def __init__(self, name: str = "default_agent"):
        self.name = name
        self.llm_router = SmartLLMRouter()
        self.task_runner = TaskRunner(llm_router=self.llm_router)
        self.workflow_engine = WorkflowEngine()
        self.memory: Dict[str, Any] = {}
        logging.info(f"AutonomousAgent '{name}' initialized")

    def think(self, prompt: str, prefer_bitnet: bool = False) -> str:
        result = self.llm_router.route(prompt, prefer_bitnet=prefer_bitnet)
        return result.get("response", "")

    def do_task(self, task_type: str, prompt: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        return self.task_runner.run_task(task_type, prompt, prefer_bitnet=prefer_bitnet)

    def remember(self, key: str, value: Any):
        self.memory[key] = value

    def recall(self, key: str) -> Any:
        return self.memory.get(key)

    def register_workflow(self, name: str, steps: List[Callable]):
        self.workflow_engine.register_workflow(name, steps)

    def run_workflow(self, name: str, initial_input: Any = None) -> List[Dict[str, Any]]:
        return self.workflow_engine.run(name, initial_input)

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ollama_available": self.llm_router.ollama_available,
            "bitnet_available": self.llm_router.bitnet_available,
            "memory_keys": list(self.memory.keys()),
            "workflows": self.workflow_engine.list_workflows(),
        }
