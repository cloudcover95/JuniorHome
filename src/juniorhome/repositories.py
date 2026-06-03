# path: src/juniorhome/repositories.py
#!/usr/bin/env python3
"""
Repositories (Domain-Driven Design)

Repository pattern for the Second Brain / Data Lake.
Provides a clean abstraction over persistence.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class IFindingsRepository(ABC):
    @abstractmethod
    def save(self, finding: Dict[str, Any]):
        pass

    @abstractmethod
    def find_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def find_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        pass


class FindingsRepository(IFindingsRepository):
    """
    Concrete repository backed by DataLakeIntegration.
    """

    def __init__(self, datalake_integration):
        self.datalake = datalake_integration

    def save(self, finding: Dict[str, Any]):
        self.datalake.store_finding(finding)

    def find_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.datalake.get_recent_findings(limit=limit)

    def find_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        all_findings = self.datalake.get_recent_findings(limit=1000)
        return [f for f in all_findings if topic.lower() in str(f).lower()]
