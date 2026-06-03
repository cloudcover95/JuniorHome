# path: src/juniorhome/autonomous_agent.py
#!/usr/bin/env python3
"""
Autonomous Agent (Deep Integration)

Now cross-pollinated with JuniorLLM's AutonomousCoder for code generation
and self-modification capabilities, while respecting all efficiency layers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    from juniorllm.autonomy.autonomous_coder import AutonomousCoder
    HAS_JUNIORLLM = True
except ImportError:
    HAS_JUNIORLLM = False
    AutonomousCoder = None

from .smart_llm_router import SmartLLMRouter
from .task_runner import TaskRunner
from .workflow_engine import WorkflowEngine
from .quantized_model_manager import QuantizedModelManager
from .edge_compute_manager import EdgeComputeManager
from .edge_runtime import EdgeRuntime

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AutonomousAgent:
    def __init__(self, name: str = "default_agent"):
        self.name = name
        self.llm_router = SmartLLMRouter()
        self.task_runner = TaskRunner(llm_router=self.llm_router)
        self.workflow_engine = WorkflowEngine()
        self.quantized_models = QuantizedModelManager()
        self.edge_manager = EdgeComputeManager(prefer_quantized=True)
        self.edge_runtime = EdgeRuntime()

        # Cross-pollination with JuniorLLM
        if HAS_JUNIORLLM and AutonomousCoder:
            self.autonomous_coder = AutonomousCoder()
        else:
            self.autonomous_coder = None

        self.memory: Dict[str, Any] = {}
        logging.info(f"AutonomousAgent '{name}' initialized with full cross-pollination")

    def think(self, prompt: str, prefer_bitnet: bool = False, task_complexity: str = "medium") -> str:
        use_quantized = self.edge_manager.should_use_quantized(task_complexity)

        def _do_think():
            result = self.llm_router.route(prompt, prefer_bitnet=prefer_bitnet or use_quantized)
            return result.get("response", "")

        return self.edge_runtime.execute_efficiently(
            _do_think,
            task_name="think",
            task_complexity=task_complexity,
        )

    def propose_code_change(self, task: str) -> Dict[str, Any]:
        if not self.autonomous_coder:
            return {"error": "JuniorLLM AutonomousCoder not available"}

        proposal = self.autonomous_coder.propose_code_change(task)
        return proposal

    def generate_and_review_code(self, task: str) -> Dict[str, Any]:
        if not self.autonomous_coder:
            return {"error": "JuniorLLM not available"}

        generation = self.autonomous_coder.code_generator.generate_code(task)
        review = self.autonomous_coder.code_generator.review_code(generation.get("code", ""))
        return {"generation": generation, "review": review}

    def do_task(self, task_type: str, prompt: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        return self.task_runner.run_task(task_type, prompt, prefer_bitnet=prefer_bitnet)

    def load_quantized_gemma(self, model_name: str = "google/gemma-2-2b") -> bool:
        return self.quantized_models.load_gemma_ternary(model_name)

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
            "has_autonomous_coder": self.autonomous_coder is not None,
            "memory_keys": list(self.memory.keys()),
            "workflows": self.workflow_engine.list_workflows(),
        }
