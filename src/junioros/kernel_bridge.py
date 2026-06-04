# path: src/juniorhome/junioros/kernel_bridge.py
#!/usr/bin/env python3
"""
JuniorOSKernelBridge (v128 - Structured Ring Buffer)

Improved with basic circular ring buffer behavior and richer
state persistence for critical architecture and operational data.
"""

import logging
import mmap
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")

DEVICE_PATH = "/dev/junior_spark"
RING_SIZE = 1048576  # 1MB


@dataclass
class KernelPayload:
    ternary_data: bytes
    metadata: Dict[str, Any]
    coherence: float
    timestamp: float = 0.0


class JuniorOSKernelBridge:
    def __init__(self):
        self.device_path = DEVICE_PATH
        self._mmap = None
        self._fd = None
        self._write_offset = 0
        self._available = self._check_device()

        if self._available:
            logging.info(f"JuniorOS kernel device detected: {self.device_path}")

    def _check_device(self) -> bool:
        return os.path.exists(self.device_path) and os.access(self.device_path, os.W_OK)

    def is_available(self) -> bool:
        return self._available

    def write_ternary_manifold(
        self,
        ternary_tensor: Any,
        metadata: Optional[Dict[str, Any]] = None,
        coherence: float = 0.0,
    ) -> bool:
        if not self._available:
            logging.debug("Kernel device not available. Skipping write.")
            return False

        try:
            arr = np.asarray(ternary_tensor, dtype=np.int8)
            byte_data = arr.tobytes()

            if self._mmap is None:
                self._try_open_mmap()

            payload_size = len(byte_data)

            if self._mmap is not None:
                # Simple ring buffer behavior
                if self._write_offset + payload_size > RING_SIZE:
                    self._write_offset = 0

                self._mmap.seek(self._write_offset)
                self._mmap.write(byte_data)
                self._write_offset += payload_size
                logging.debug(f"Wrote to kernel ring buffer at offset {self._write_offset}")
                return True

            with open(self.device_path, "wb") as f:
                f.write(byte_data)
            return True

        except Exception as e:
            logging.warning(f"Kernel write failed: {e}")
            return False

    def _try_open_mmap(self):
        try:
            self._fd = os.open(self.device_path, os.O_RDWR)
            self._mmap = mmap.mmap(self._fd, RING_SIZE, access=mmap.ACCESS_WRITE)
        except Exception as e:
            logging.debug(f"mmap not available: {e}")
            self._mmap = None

    def close(self):
        if self._mmap:
            try:
                self._mmap.close()
            except:
                pass
            self._mmap = None
        if self._fd:
            try:
                os.close(self._fd)
            except:
                pass
            self._fd = None

    def __del__(self):
        self.close()
