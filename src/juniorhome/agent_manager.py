# path: src/juniorhome/agent_manager.py
#!/usr/bin/env python3
"""
JuniorHome Agent Manager

Manages multiple agents (swarm instances, BitNet reasoning sessions, etc).
Production scaffolding for multi-agent orchestration.
"""

import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class AgentManager:
    """
    Manages registration and lifecycle of agents in the orchestrator.
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        logging.info("AgentManager initialized")

    def register_agent(self, name: str, agent: Any) -> None:
        self.agents[name] = agent
        logging.info(f"Agent registered: {name}")

    def get_agent(self, name: str) -> Optional[Any]:
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def remove_agent(self, name: str) -> bool:
        if name in self.agents:
            del self.agents[name]
            logging.info(f"Agent removed: {name}")
            return True
        return False
