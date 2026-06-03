# path: src/juniorhome/use_cases.py
#!/usr/bin/env python3
"""
Use Cases / Application Services

High-level use cases that orchestrate domain logic.
This layer sits above the domain/services and below the presentation.
"""

import logging
from typing import Any, Dict, List

from .interfaces import ISecondBrain

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ProcessVaultUseCase:
    """
    Use case: Process the Obsidian vault and store important findings.
    """

    def __init__(self, second_brain: ISecondBrain):
        self.second_brain = second_brain

    def execute(self) -> List[Dict[str, Any]]:
        logging.info("Executing ProcessVaultUseCase...")
        findings = self.second_brain.process_vault()

        # Store important ones
        important = [f for f in findings if "High" in str(f.get("llm_response", ""))]
        for finding in important:
            self.second_brain.store_finding(finding)

        logging.info(f"Processed vault. Stored {len(important)} important findings.")
        return important


class QuerySecondBrainUseCase:
    """
    Use case: Query the Second Brain.
    """

    def __init__(self, second_brain: ISecondBrain):
        self.second_brain = second_brain

    def execute(self, topic: str) -> List[Dict[str, Any]]:
        return self.second_brain.query(topic)
