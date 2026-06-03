# path: src/juniorhome/autonomous_agent.py
#!/usr/bin/env python3
"""
Autonomous Agent (Edge Efficient)

Now integrates EdgeComputeManager for automatic efficient routing
between quantized and full-precision paths.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .smart_llm_router import SmartLLMRouter
from .task_runner import TaskRunner
from .workflow_engine import WorkflowEngine
from .quantized_model_manager import QuantizedModelManager
from .edge_compute_manager import EdgeComputeManager

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AutonomousAgent:
    def __init__(self, name: str = "default_agent"):
        self.name = name
        self.llm_router = SmartLLMRouter()
        self.task_runner = TaskRunner(llm_router=self.llm_router)
        self.workflow_engine = WorkflowEngine()
        self.quantized_models = QuantizedModelManager()
        self.edge_manager = EdgeComputeManager(prefer_quantized=True)
        self.memory: Dict[str, Any] = {}
        logging.info(f"AutonomousAgent '{name}' initialized (edge efficient)")

    def think(self, prompt: str, prefer_bitnet: bool = False, task_complexity: str = "medium") -> str:
        use_quantized = self.edge_manager.should_use_quantized(task_complexity)

        def _do_think():
            result = self.llm_router.route(prompt, prefer_bitnet=prefer_bitnet or use_quantized)
            return result.get("response", "")

        return self.edge_manager.execute_efficiently(
            _do_think,
            use_quantized=use_quantized,
            task_complexity=task_complexity,
        )

    def do_task(self, task_type: str, prompt: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        return self.task_runner.run_task(task_type, prompt, prefer_bitnet=prefer_bitnet)

    def load_quantized_gemma(self, model_name: str = "google/gemma-2-2b") -> bool:
        return self.quantized_models.load_gemma_ternary(model_name)

    def get_quantized_model(self, name: str) -> Optional[Any]:
        return self.quantized_models.get_model(name)

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
            "quantized_models_loaded": self.quantized_models.list_loaded_models(),
            "edge_stats": self.edge_manager.get_stats(),
            "memory_keys": list(self.memory.keys()),
            "workflows": self.workflow_engine.list_workflows(),
        }
