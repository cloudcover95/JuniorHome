# path: src/juniorhome/interfaces.py
#!/usr/bin/env python3
"""
Core Interfaces

Defines abstract interfaces for major components.
This enables better dependency inversion and cleaner architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IDataLake(ABC):
    @abstractmethod
    def write(self, table_name: str, data: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def read(self, table_name: str) -> Any:
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        pass


class ISecondBrain(ABC):
    @abstractmethod
    def process_vault(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def store_finding(self, finding: Dict[str, Any]):
        pass

    @abstractmethod
    def get_recent_findings(self, limit: int = 100) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def query(self, topic: str) -> List[Dict[str, Any]]:
        pass


class IOrchestrator(ABC):
    @abstractmethod
    def status(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def route_llm(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass


class IKnowledgeService(ABC):
    @abstractmethod
    def process_vault_once(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def start_monitoring(self):
        pass
