# path: src/juniorhome/sandbox_executor.py
#!/usr/bin/env python3
"""
Sandbox Executor

Safe execution environment for actions suggested by LLMs or agents.
Integrates with LLMSecurityGuard to validate and safely run operations.
"""

import logging
from typing import Any, Callable, Dict, Optional

from .llm_security_guard import LLMSecurityGuard

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SandboxExecutor:
    """
    Executes actions in a controlled, validated environment.
    """

    def __init__(self, security_guard: Optional[LLMSecurityGuard] = None):
        self.security_guard = security_guard or LLMSecurityGuard(strict_mode=True)
        logging.info("SandboxExecutor initialized")

    def execute_safely(
        self,
        action: Callable[[], Any],
        llm_output: str,
        require_user_confirmation: bool = True,
    ) -> Dict[str, Any]:
        """
        Validates LLM output and executes the action only if safe.
        """
        validation = self.security_guard.validate_output(llm_output)

        if not validation["safe"]:
            return {
                "executed": False,
                "reason": "Unsafe LLM output detected",
                "validation": validation,
            }

        if require_user_confirmation:
            logging.warning("User confirmation required for this action (not implemented in this version)")

        try:
            result = action()
            return {
                "executed": True,
                "result": result,
                "validation": validation,
            }
        except Exception as e:
            logging.error(f"Sandbox execution failed: {e}")
            return {
                "executed": False,
                "error": str(e),
            }

    def register_safe_action(self, name: str, action: Callable):
        # Placeholder for future registry of pre-approved safe actions
        logging.info(f"Safe action registered: {name}")
        return action
