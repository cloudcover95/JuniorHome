# path: src/juniorhome/datalake_manager.py
#!/usr/bin/env python3
"""
Data Lake Manager

Production-grade multi-backend data lake for JuniorHome.
Supports Parquet (default), SQLite, and basic Markdown/Obsidian integration.
Designed for both structured data and unstructured knowledge streams.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class DataLakeManager:
    """
    Unified data lake supporting multiple storage backends.
    """

    def __init__(self, base_path: str, default_backend: str = "parquet"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.default_backend = default_backend
        self.sqlite_conn: Optional[sqlite3.Connection] = None

        logging.info(f"DataLakeManager initialized at {self.base_path} (backend={default_backend})")

    def _get_parquet_path(self, table_name: str) -> Path:
        return self.base_path / f"{table_name}.parquet"

    def write(self, table_name: str, data: List[Dict[str, Any]], backend: Optional[str] = None):
        backend = backend or self.default_backend

        if backend == "parquet":
            table = pa.Table.from_pydict({k: [d[k] for d in data] for k in data[0].keys()})
            pq.write_table(table, self._get_parquet_path(table_name), compression="ZSTD")

        elif backend == "sqlite":
            if self.sqlite_conn is None:
                self.sqlite_conn = sqlite3.connect(str(self.base_path / "datalake.db"))

            if not data:
                return

            columns = list(data[0].keys())
            placeholders = ", ".join(["?"] * len(columns))
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

            self.sqlite_conn.execute(create_sql)
            self.sqlite_conn.executemany(insert_sql, [tuple(d.values()) for d in data])
            self.sqlite_conn.commit()

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logging.info(f"Wrote {len(data)} rows to {table_name} ({backend})")

    def read(self, table_name: str, backend: Optional[str] = None) -> Any:
        backend = backend or self.default_backend

        if backend == "parquet":
            path = self._get_parquet_path(table_name)
            if not path.exists():
                return None
            return pq.read_table(path)

        elif backend == "sqlite":
            if self.sqlite_conn is None:
                self.sqlite_conn = sqlite3.connect(str(self.base_path / "datalake.db"))

            try:
                cursor = self.sqlite_conn.execute(f"SELECT * FROM {table_name}")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            except sqlite3.OperationalError:
                return []

        return None

    def list_tables(self) -> List[str]:
        tables = []
        for f in self.base_path.glob("*.parquet"):
            tables.append(f.stem)

        if (self.base_path / "datalake.db").exists():
            if self.sqlite_conn is None:
                self.sqlite_conn = sqlite3.connect(str(self.base_path / "datalake.db"))
            cursor = self.sqlite_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables.extend([row[0] for row in cursor.fetchall()])

        return sorted(set(tables))
