# path: src/juniorhome/training_data_hook.py
#!/usr/bin/env python3
"""
Training Data Hook

Utilities for preparing and exporting data from JuniorHome
for model fine-tuning and continued pretraining.
Supports JSONL, Parquet, and Markdown formats suitable for LLM training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TrainingDataHook:
    """
    Prepares and exports data for model training.
    """

    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"TrainingDataHook initialized at {self.output_dir}")

    def export_jsonl(self, data: List[Dict[str, Any]], filename: str = "train.jsonl"):
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(f"Exported {len(data)} records to {path}")
        return str(path)

    def export_parquet(self, data: List[Dict[str, Any]], filename: str = "train.parquet"):
        if not data:
            return None
        table = pa.Table.from_pydict({k: [d[k] for d in data] for k in data[0].keys()})
        path = self.output_dir / filename
        pq.write_table(table, path, compression="ZSTD")
        logging.info(f"Exported {len(data)} records to {path}")
        return str(path)

    def export_markdown_dataset(self, conversations: List[Dict[str, str]], filename: str = "train.md"):
        """
        Export in a simple markdown format useful for some fine-tuning pipelines.
        """
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(f"## {conv.get('title', 'Conversation')}\n\n")
                f.write(f"**User:** {conv.get('user', '')}\n\n")
                f.write(f"**Assistant:** {conv.get('assistant', '')}\n\n---\n\n")
        logging.info(f"Exported markdown dataset to {path}")
        return str(path)

    def prepare_from_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert internal knowledge items into training-friendly format.
        """
        training_data = []
        for item in knowledge_items:
            training_data.append({
                "instruction": item.get("prompt", ""),
                "input": item.get("context", ""),
                "output": item.get("response", ""),
            })
        return training_data
