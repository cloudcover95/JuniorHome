# path: src/juniorhome/knowledge_service.py
#!/usr/bin/env python3
"""
Knowledge Service

High-level service that combines the ResilientKnowledgePipeline,
DataLakeIntegration, Scheduler, and other components into an easy-to-use
service for automated knowledge processing from Obsidian and other sources.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .resilient_knowledge_pipeline import ResilientKnowledgePipeline
from .datalake_integration import DataLakeIntegration
from .scheduler import Scheduler

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class KnowledgeService:
    """
    High-level service for automated knowledge processing.
    """

    def __init__(
        self,
        vault_path: str,
        use_ternary: bool = False,
        enable_scheduling: bool = True,
    ):
        self.pipeline = ResilientKnowledgePipeline(
            vault_path=vault_path,
            use_ternary=use_ternary,
            on_important_finding=self._on_important_finding,
        )
        self.datalake = DataLakeIntegration()
        self.scheduler = Scheduler() if enable_scheduling else None

        if self.scheduler:
            self.scheduler.add_daily_task(
                "process_obsidian_vault",
                self.process_vault_once,
                hour=3,
                minute=0,
            )
            self.scheduler.start()

        logging.info("KnowledgeService initialized")

    def _on_important_finding(self, file_path: str, assessment: str):
        finding = {
            "file": file_path,
            "assessment": assessment,
            "timestamp": __import__("time").time(),
        }
        self.datalake.store_finding(finding, table="important_findings")
        logging.info(f"Important finding stored from {file_path}")

    def process_vault_once(self) -> List[Dict[str, Any]]:
        logging.info("Processing Obsidian vault...")
        results = self.pipeline.process_once()
        logging.info(f"Processed {len(results)} files")
        return results

    def start_monitoring(self):
        logging.info("Starting real-time monitoring of Obsidian vault...")
        self.pipeline.start_watching()

    def get_recent_important_findings(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.datalake.get_recent_findings(table="important_findings", limit=limit)

    def shutdown(self):
        if self.scheduler:
            self.scheduler.stop()
        logging.info("KnowledgeService shutdown complete")
