# path: src/juniorllm/memory/memory_backend.py

"""
MemoryBackend Interface

Abstract interface for pluggable memory backends.

Current: InMemoryBackend (default)
Future: JuniorMemSysBackend (persistent + topological)

This enables clean separation between the active SHEEPMemory logic
and long-term storage / querying.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryBackend(ABC):
    @abstractmethod
    def store_awakening(self, record: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_history(self, last_n: int = 50) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def store_consolidated_insights(self, insights: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_consolidated_insights(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def store_performance(self, profile: str, value: float) -> None:
        pass

    @abstractmethod
    def get_performance(self, profile: str) -> float:
        pass

    @abstractmethod
    def store_lifecycle(self, profile: str, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_lifecycle(self, profile: str) -> Dict[str, Any]:
        pass


class InMemoryBackend(MemoryBackend):
    """Default in-memory implementation (current behavior)."""

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._consolidated_insights: Dict[str, Any] = {}
        self._performance: Dict[str, float] = {}
        self._lifecycle: Dict[str, Dict[str, Any]] = {}

    def store_awakening(self, record: Dict[str, Any]) -> None:
        self._history.append(record)

    def get_history(self, last_n: int = 50) -> List[Dict[str, Any]]:
        return self._history[-last_n:]

    def store_consolidated_insights(self, insights: Dict[str, Any]) -> None:
        self._consolidated_insights = insights

    def get_consolidated_insights(self) -> Dict[str, Any]:
        return self._consolidated_insights.copy()

    def store_performance(self, profile: str, value: float) -> None:
        self._performance[profile] = value

    def get_performance(self, profile: str) -> float:
        return self._performance.get(profile, 0.0)

    def store_lifecycle(self, profile: str, data: Dict[str, Any]) -> None:
        self._lifecycle[profile] = data

    def get_lifecycle(self, profile: str) -> Dict[str, Any]:
        return self._lifecycle.get(profile, {}).copy()
