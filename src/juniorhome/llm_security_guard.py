# path: src/juniorhome/llm_security_guard.py
#!/usr/bin/env python3
"""
LLM Security Guard

Production-grade security layer for local LLMs and autonomous agents.
Provides input sanitization, output validation, prompt injection detection,
and safe execution wrappers.

Designed for sovereign, air-gapped systems where LLM outputs
can trigger real actions (workflows, visualizations, execution).
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class LLMSecurityGuard:
    """
    Security guard for LLM interactions and agent actions.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.blocked_patterns = [
            r"(?i)(ignore previous instructions|disregard|forget everything)",
            r"(?i)(execute|run|system|shell|rm -rf|sudo)",
            r"(?i)(delete|format|wipe|destroy) (all|everything|data)",
        ]
        logging.info(f"LLMSecurityGuard initialized (strict={strict_mode})")

    def sanitize_prompt(self, prompt: str) -> str:
        """
        Basic sanitization of user/LLM prompts.
        """
        cleaned = prompt.strip()

        # Remove obvious injection attempts in strict mode
        if self.strict_mode:
            for pattern in self.blocked_patterns:
                if re.search(pattern, cleaned):
                    logging.warning("Potential prompt injection detected and sanitized")
                    cleaned = re.sub(pattern, "[BLOCKED]", cleaned)

        return cleaned

    def validate_output(self, output: str) -> Dict[str, Any]:
        """
        Validates LLM output before allowing execution.
        """
        issues = []

        for pattern in self.blocked_patterns:
            if re.search(pattern, output):
                issues.append(f"Unsafe pattern detected: {pattern}")

        is_safe = len(issues) == 0

        return {
            "safe": is_safe,
            "issues": issues,
            "original_output": output,
        }

    def safe_execute(
        self,
        action: Callable[[], Any],
        llm_output: str,
        require_confirmation: bool = True,
    ) -> Dict[str, Any]:
        """
        Safely executes an action only after validating LLM output.
        """
        validation = self.validate_output(llm_output)

        if not validation["safe"]:
            return {
                "executed": False,
                "reason": "Unsafe LLM output",
                "validation": validation,
            }

        if require_confirmation:
            # In a real system this could prompt the user or require explicit approval
            logging.info("Action requires confirmation (simulated in this version)")

        try:
            result = action()
            return {
                "executed": True,
                "result": result,
                "validation": validation,
            }
        except Exception as e:
            return {
                "executed": False,
                "error": str(e),
            }

    def add_blocked_pattern(self, pattern: str):
        self.blocked_patterns.append(pattern)
        logging.info(f"Added blocked pattern: {pattern}")
