# path: src/juniorhome/datalake.py
#!/usr/bin/env python3
"""
JuniorHome Data Lake Layer

Production-grade Parquet-based data lake for the sovereign edge stack.
Supports time-series and structured data with efficient appends.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class DataLake:
    """
    Simple but production-oriented Parquet data lake.
    Designed for edge use with good compression and append support.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(self, table_name: str, data: List[Dict[str, Any]], partition: Optional[str] = None) -> Path:
        if not data:
            raise ValueError("Cannot write empty data")

        table = pa.Table.from_pydict({k: [d[k] for d in data] for k in data[0].keys()})

        if partition:
            file_path = self.base_path / partition / f"{table_name}.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path = self.base_path / f"{table_name}.parquet"

        pq.write_table(table, file_path, compression="ZSTD")
        logging.info(f"Wrote {len(data)} rows to {file_path}")
        return file_path

    def read(self, table_name: str, partition: Optional[str] = None) -> Optional[pa.Table]:
        if partition:
            file_path = self.base_path / partition / f"{table_name}.parquet"
        else:
            file_path = self.base_path / f"{table_name}.parquet"

        if not file_path.exists():
            return None

        return pq.read_table(file_path)
