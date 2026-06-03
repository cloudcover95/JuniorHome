# path: src/juniorhome/second_brain.py
#!/usr/bin/env python3
"""
Second Brain

High-level service representing the Obsidian-backed Data Lake as a
persistent "Second Brain" for the sovereign agent ecosystem.

Uses Service Locator / Container pattern for clean architecture.
"""

import logging
from typing import Any, Dict, List, Optional

from .datalake_manager import DataLakeManager
from .datalake_integration import DataLakeIntegration
from .resilient_knowledge_pipeline import ResilientKnowledgePipeline

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class SecondBrain:
    """
    The persistent knowledge layer (Obsidian + Data Lake).
    Acts as the long-term memory / Second Brain for agents.
    """

    def __init__(self, vault_path: str, data_dir: str = "data"):
        self.vault_path = vault_path
        self.datalake = DataLakeManager(base_path=data_dir)
        self.integration = DataLakeIntegration(datalake=self.datalake)
        self.pipeline = ResilientKnowledgePipeline(vault_path=vault_path)

        logging.info(f"SecondBrain initialized (vault={vault_path})")

    def process_vault(self) -> List[Dict[str, Any]]:
        return self.pipeline.process_once()

    def store_finding(self, finding: Dict[str, Any]):
        return self.integration.store_finding(finding)

    def get_recent_findings(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.integration.get_recent_findings(limit=limit)

    def query(self, topic: str) -> List[Dict[str, Any]]:
        # Simple query against data lake (can be expanded)
        findings = self.get_recent_findings(limit=500)
        return [f for f in findings if topic.lower() in str(f).lower()]
