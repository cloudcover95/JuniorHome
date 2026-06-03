# path: src/juniorhome/workflow_engine.py
#!/usr/bin/env python3
"""
Workflow Engine

Lightweight workflow engine for defining and running automated pipelines
in JuniorHome. Supports chaining LLM calls, ternary analysis, file processing,
and custom steps with error handling.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class WorkflowEngine:
    """
    Simple but powerful workflow runner.
    """

    def __init__(self):
        self.workflows: Dict[str, List[Callable]] = {}
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        logging.info("WorkflowEngine initialized")

    def register_workflow(self, name: str, steps: List[Callable[[Any], Any]]):
        self.workflows[name] = steps
        self.results[name] = []
        logging.info(f"Workflow registered: {name} ({len(steps)} steps)")

    def run(self, name: str, initial_input: Any = None) -> List[Dict[str, Any]]:
        if name not in self.workflows:
            raise KeyError(f"Workflow not found: {name}")

        steps = self.workflows[name]
        current = initial_input
        step_results = []

        for i, step in enumerate(steps):
            try:
                start = time.time()
                output = step(current)
                duration = time.time() - start

                result = {
                    "step": i,
                    "input": current,
                    "output": output,
                    "duration": duration,
                    "success": True,
                }
                step_results.append(result)
                current = output

            except Exception as e:
                logging.error(f"Workflow {name} failed at step {i}: {e}")
                result = {
                    "step": i,
                    "input": current,
                    "error": str(e),
                    "success": False,
                }
                step_results.append(result)
                break

        self.results[name].append(step_results)
        return step_results

    def get_results(self, name: str) -> List[List[Dict[str, Any]]]:
        return self.results.get(name, [])

    def list_workflows(self) -> List[str]:
        return list(self.workflows.keys())
