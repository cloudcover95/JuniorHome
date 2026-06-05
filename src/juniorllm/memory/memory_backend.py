# path: src/juniorllm/memory/memory_backend.py

"""
MemoryBackend implementations

- InMemoryBackend: Default fast in-memory storage
- JuniorMemSysBackend: Stub for future integration with JuniorMemSys-Suite
  (the persistent topological long-term memory layer)

When ready, JuniorMemSysBackend can delegate to actual JuniorMemSys
methods for persistent .parquet storage + TDA querying.
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
    """Default fast in-memory backend."""

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
    Backend stub for integration with JuniorMemSys-Suite.

    Current behavior: Falls back to in-memory storage (same as InMemoryBackend).

    Future behavior:
    - store_awakening() → Persist to JuniorMemSys with topological metadata
    - get_history() → Query from JuniorMemSys (with optional TDA filters)
    - consolidated_insights → Store as high-level topological summaries

    This class acts as the bridge between fast SHEEPMemory reasoning
    and the persistent long-term memory system (JuniorMemSys).
    """

    def __init__(self):
        # For now, delegate to in-memory storage.
        # When JuniorMemSys is ready, replace these with real calls.
        self._fallback = InMemoryBackend()
        self._connected = False  # Flag for when real JuniorMemSys connection exists

    def store_awakening(self, record: Dict[str, Any]) -> None:
        # TODO: When integrated, add topological features (e.g., TDA signature)
        # and store via JuniorMemSys API.
        self._fallback.store_awakening(record)

    def get_history(self, last_n: int = 50) -> List[Dict[str, Any]]:
        return self._fallback.get_history(last_n)

    def store_consolidated_insights(self, insights: Dict[str, Any]) -> None:
        # TODO: Store as high-level memory object in JuniorMemSys
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

    def is_connected(self) -> bool:
        return self._connected

    # Placeholder for future real integration methods
    def connect_to_memsys(self):
        """Future method to establish connection to JuniorMemSys-Suite."""
        print("[JuniorMemSysBackend] Connection to JuniorMemSys not yet implemented.")
        self._connected = True
