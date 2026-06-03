# path: src/juniorhome/datalake_integration.py
#!/usr/bin/env python3
"""
DataLake Integration

Connects the DataLakeManager with knowledge processing pipelines.
Allows storing processed findings, assessments, and structured data
from Obsidian streams and other sources into the data lake.
"""

import logging
from typing import Any, Dict, List, Optional

from .datalake_manager import DataLakeManager

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class DataLakeIntegration:
    """
    Bridges knowledge processing with the data lake.
    """

    def __init__(self, datalake: Optional[DataLakeManager] = None):
        self.datalake = datalake or DataLakeManager(base_path="data")
        logging.info("DataLakeIntegration initialized")

    def store_finding(self, finding: Dict[str, Any], table: str = "findings"):
        """
        Store a processed finding/assessment into the data lake.
        """
        try:
            self.datalake.write(table, [finding])
            logging.info(f"Stored finding in table '{table}'")
            return True
        except Exception as e:
            logging.error(f"Failed to store finding: {e}")
            return False

    def store_batch(self, items: List[Dict[str, Any]], table: str = "batch_data"):
        if not items:
            return False
        try:
            self.datalake.write(table, items)
            logging.info(f"Stored batch of {len(items)} items in '{table}'")
            return True
        except Exception as e:
            logging.error(f"Failed to store batch: {e}")
            return False

    def get_recent_findings(self, table: str = "findings", limit: int = 100) -> List[Dict[str, Any]]:
        try:
            data = self.datalake.read(table)
            if data is None:
                return []
            # For Parquet, convert to list of dicts
            if hasattr(data, "to_pylist"):
                return data.to_pylist()[-limit:]
            return data[-limit:] if isinstance(data, list) else []
        except Exception as e:
            logging.error(f"Failed to retrieve findings: {e}")
            return []
