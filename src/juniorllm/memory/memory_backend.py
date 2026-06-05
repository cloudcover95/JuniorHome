# path: src/juniorllm/memory/memory_backend.py

"""
MemoryBackend implementations for SHEEPMemory.

This module provides the storage abstraction layer.

- InMemoryBackend: Fast default
- JuniorMemSysBackend: Designed for integration with JuniorMemSys-Suite
  (the ecosystem's persistent topological long-term memory system).

When fully integrated, JuniorMemSysBackend will delegate storage and
complex queries (TDA, persistence landscapes, etc.) to JuniorMemSys.
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
    """Default in-memory implementation. Fast and simple."""

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


class JuniorMemSysBackend(MemoryBackend):
    """
    Backend designed for integration with JuniorMemSys-Suite.

    Current state: Functional stub (delegates to in-memory for compatibility).

    Integration roadmap:
    - When JuniorMemSys is ready, replace internal storage with calls to
      JuniorMemSys methods for persistent storage and topological queries.
    - SHEEPMemory can then benefit from long-term memory, TDA-based retrieval,
      and cross-component memory sharing.

    This class serves as the official bridge between the fast reasoning
    memory (SHEEPMemory) and the ecosystem's long-term memory system.
    """

    def __init__(self):
        self._fallback = InMemoryBackend()
        self._memsys_instance = None  # Will hold real JuniorMemSys connection
        self._connected = False

    def store_awakening(self, record: Dict[str, Any]) -> None:
        # TODO: When integrated:
        #   - Add topological features (e.g. persistence diagram summary)
        #   - Call self._memsys_instance.store_memory(record, metadata=...)
        self._fallback.store_awakening(record)

    def get_history(self, last_n: int = 50) -> List[Dict[str, Any]]:
        # TODO: Replace with topological query from JuniorMemSys when available
        return self._fallback.get_history(last_n)

    def store_consolidated_insights(self, insights: Dict[str, Any]) -> None:
        # TODO: Store as high-level consolidated memory object in JuniorMemSys
        self._fallback.store_consolidated_insights(insights)

    def get_consolidated_insights(self) -> Dict[str, Any]:
        return self._fallback.get_consolidated_insights()

    def store_performance(self, profile: str, value: float) -> None:
        self._fallback.store_performance(profile, value)

    def get_performance(self, profile: str) -> float:
        return self._fallback.get_performance(profile)

    def store_lifecycle(self, profile: str, data: Dict[str, Any]) -> None:
        self._fallback.store_lifecycle(profile, data)

    def get_lifecycle(self, profile: str) -> Dict[str, Any]:
        return self._fallback.get_lifecycle(profile)

    # === Integration helper methods ===

    def connect_to_memsys(self, memsys_instance: Any = None):
        """Establish connection to a JuniorMemSys instance."""
        self._memsys_instance = memsys_instance
        self._connected = True
        print("[JuniorMemSysBackend] Connected to JuniorMemSys (stub mode).")

    def is_connected(self) -> bool:
        return self._connected

    def persist_to_memsys(self):
        """Future method: Flush current state to JuniorMemSys for long-term storage."""
        if not self._connected:
            print("[JuniorMemSysBackend] Not connected to JuniorMemSys yet.")
            return
        # TODO: Implement real persistence logic here
        print("[JuniorMemSysBackend] Persisting to JuniorMemSys... (not yet implemented)")
