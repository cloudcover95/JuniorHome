# path: src/juniorhome/task_runner.py
#!/usr/bin/env python3
"""
Task Runner

Simple but powerful task runner for common workflows.
Uses SmartLLMRouter for LLM calls and supports ternary analysis
when needed. Designed for real daily use.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .smart_llm_router import SmartLLMRouter

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TaskRunner:
    """
    Executes common tasks using the LLM router and optional ternary analysis.
    """

    def __init__(self, llm_router: Optional[SmartLLMRouter] = None):
        self.llm_router = llm_router or SmartLLMRouter()
        self.task_history: List[Dict[str, Any]] = []
        logging.info("TaskRunner initialized")

    def run_task(
        self,
        task_type: str,
        prompt: str,
        prefer_bitnet: bool = False,
        model: str = "llama3.2",
    ) -> Dict[str, Any]:
        """
        Run a named task type with the LLM router.
        """
        result = self.llm_router.route(
            prompt=prompt,
            prefer_bitnet=prefer_bitnet,
            model=model,
        )

        task_record = {
            "task_type": task_type,
            "prompt": prompt,
            "result": result,
            "used_bitnet": result.get("backend") == "bitnet-mlx",
        }
        self.task_history.append(task_record)

        return result

    def research(self, topic: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        prompt = f"Research this topic thoroughly and summarize key findings: {topic}"
        return self.run_task("research", prompt, prefer_bitnet=prefer_bitnet)

    def analyze_code(self, code: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        prompt = f"Analyze this code and provide feedback:\n\n{code}"
        return self.run_task("code_analysis", prompt, prefer_bitnet=prefer_bitnet)

    def summarize(self, text: str, prefer_bitnet: bool = False) -> Dict[str, Any]:
        prompt = f"Summarize the following text concisely:\n\n{text}"
        return self.run_task("summarize", prompt, prefer_bitnet=prefer_bitnet)

    def get_history(self) -> List[Dict[str, Any]]:
        return self.task_history

    def clear_history(self):
        self.task_history.clear()
