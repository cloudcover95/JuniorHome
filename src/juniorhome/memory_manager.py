# path: src/juniorhome/memory_manager.py
#!/usr/bin/env python3
"""
Memory Manager

Optimized memory allocation strategies for edge inference and
Second Brain operations. Focuses on low-memory, long-running sovereign systems.
"""

import logging
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class MemoryManager:
    """
    Manages memory allocation with edge efficiency in mind.
    """

    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.allocated: Dict[str, int] = {}
        self.total_allocated = 0
        logging.info(f"MemoryManager initialized (max={max_memory_mb}MB)")

    def allocate(self, key: str, size_mb: int) -> bool:
        if self.total_allocated + size_mb > self.max_memory_mb:
            logging.warning(f"Memory allocation denied for {key} (would exceed limit)")
            return False

        self.allocated[key] = size_mb
        self.total_allocated += size_mb
        logging.debug(f"Allocated {size_mb}MB for {key}")
        return True

    def release(self, key: str):
        if key in self.allocated:
            self.total_allocated -= self.allocated[key]
            del self.allocated[key]
            logging.debug(f"Released memory for {key}")

    def get_usage(self) -> Dict[str, Any]:
        return {
            "total_allocated_mb": self.total_allocated,
            "max_mb": self.max_memory_mb,
            "usage_percent": (self.total_allocated / self.max_memory_mb) * 100,
            "allocations": self.allocated.copy(),
        }

    def can_allocate(self, size_mb: int) -> bool:
        return self.total_allocated + size_mb <= self.max_memory_mb
