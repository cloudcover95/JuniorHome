# path: src/juniorhome/policy_engine.py
#!/usr/bin/env python3
"""
Policy Engine

Allows defining and enforcing rules for what LLMs and autonomous agents
can and cannot do. Essential for production-grade control and safety.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class PolicyEngine:
    """
    Rule-based policy engine for controlling agent and LLM behavior.
    """

    def __init__(self):
        self.policies: Dict[str, List[Callable[[Any], bool]]] = {}
        logging.info("PolicyEngine initialized")

    def add_policy(self, name: str, rule: Callable[[Any], bool]):
        if name not in self.policies:
            self.policies[name] = []
        self.policies[name].append(rule)
        logging.info(f"Policy added: {name}")

    def check_action(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        all_passed = True

        for policy_name, rules in self.policies.items():
            policy_passed = True
            for rule in rules:
                try:
                    if not rule(action_context):
                        policy_passed = False
                        all_passed = False
                        break
                except Exception as e:
                    logging.error(f"Error evaluating policy {policy_name}: {e}")
                    policy_passed = False
                    all_passed = False
                    break

            results[policy_name] = policy_passed

        return {
            "allowed": all_passed,
            "policy_results": results,
        }

    def add_default_safety_policies(self):
        def no_destructive_actions(context):
            action = str(context.get("action", "")).lower()
            return not any(x in action for x in ["delete", "remove", "format", "destroy", "rm "])

        def no_system_commands(context):
            action = str(context.get("action", "")).lower()
            return not any(x in action for x in ["sudo", "shell", "exec", "system"])

        self.add_policy("no_destructive_actions", no_destructive_actions)
        self.add_policy("no_system_commands", no_system_commands)
        logging.info("Default safety policies added")
