# path: src/juniorhome/security_middleware.py
#!/usr/bin/env python3
"""
Security Middleware

Central security layer that combines LLMSecurityGuard, SandboxExecutor,
PolicyEngine, and ActionAuditor into one easy-to-use middleware.

This is the recommended way to secure LLM and agent interactions in production.
"""

import logging
from typing import Any, Callable, Dict, Optional

from .llm_security_guard import LLMSecurityGuard
from .sandbox_executor import SandboxExecutor
from .policy_engine import PolicyEngine
from .action_auditor import ActionAuditor

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SecurityMiddleware:
    """
    Central security middleware for LLM and agent operations.
    """

    def __init__(self, strict_mode: bool = True):
        self.guard = LLMSecurityGuard(strict_mode=strict_mode)
        self.sandbox = SandboxExecutor(security_guard=self.guard)
        self.policy = PolicyEngine()
        self.auditor = ActionAuditor()

        self.policy.add_default_safety_policies()
        logging.info("SecurityMiddleware initialized")

    def secure_llm_call(self, prompt: str, backend: str = "ollama") -> Dict[str, Any]:
        sanitized = self.guard.sanitize_prompt(prompt)

        # In real usage, this would call the actual LLM
        # For now we just log and return the sanitized prompt
        self.auditor.log_llm_interaction(
            prompt=sanitized,
            response="[SIMULATED]",
            backend=backend,
            success=True,
        )

        return {
            "sanitized_prompt": sanitized,
            "backend": backend,
        }

    def secure_action(
        self,
        action: Callable[[], Any],
        llm_output: str,
        action_description: str = "LLM suggested action",
    ) -> Dict[str, Any]:
        # Check policy first
        policy_result = self.policy.check_action({"action": action_description})

        if not policy_result["allowed"]:
            self.auditor.log_action(
                action_type="blocked_by_policy",
                details={"action": action_description, "policy_result": policy_result},
                success=False,
            )
            return {
                "executed": False,
                "reason": "Blocked by policy",
                "policy_result": policy_result,
            }

        # Then use sandbox for execution
        result = self.sandbox.execute_safely(action, llm_output)

        self.auditor.log_sandbox_execution(
            action_description=action_description,
            success=result.get("executed", False),
            validation_result=result.get("validation"),
        )

        return result

    def get_security_status(self) -> Dict[str, Any]:
        return {
            "guard_strict_mode": self.guard.strict_mode,
            "policies_loaded": list(self.policy.policies.keys()),
        }
