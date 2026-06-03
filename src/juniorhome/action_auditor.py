# path: src/juniorhome/action_auditor.py
#!/usr/bin/env python3
"""
Action Auditor

Production-grade auditing system for tracking LLM and agent actions.
Logs all significant operations for security review, debugging, and compliance.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ActionAuditor:
    """
    Records and stores actions taken by LLMs and autonomous agents.
    """

    def __init__(self, log_file: str = "audit.log"):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"ActionAuditor initialized. Logging to {self.log_path}")

    def log_action(
        self,
        action_type: str,
        actor: str = "llm",
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ):
        entry = {
            "timestamp": time.time(),
            "action_type": action_type,
            "actor": actor,
            "success": success,
            "details": details or {},
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logging.debug(f"Logged action: {action_type}")
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")

    def log_llm_interaction(
        self,
        prompt: str,
        response: str,
        backend: str = "unknown",
        success: bool = True,
    ):
        self.log_action(
            action_type="llm_interaction",
            actor="llm",
            details={
                "prompt": prompt[:500],  # Truncate long prompts
                "response": response[:500],
                "backend": backend,
            },
            success=success,
        )

    def log_workflow_execution(
        self,
        workflow_name: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.log_action(
            action_type="workflow_execution",
            actor="workflow_engine",
            details={
                "workflow": workflow_name,
                **(details or {}),
            },
            success=success,
        )

    def log_sandbox_execution(
        self,
        action_description: str,
        success: bool,
        validation_result: Optional[Dict[str, Any]] = None,
    ):
        self.log_action(
            action_type="sandbox_execution",
            actor="sandbox_executor",
            details={
                "action": action_description,
                "validation": validation_result,
            },
            success=success,
        )
