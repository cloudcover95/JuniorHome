# path: src/juniorhome/junioros/kernel_bridge.py
#!/usr/bin/env python3
"""
JuniorOSKernelBridge (More Robust)

Improved error handling, better logging, and clearer separation
of mmap vs regular writes. Ready for deeper production use.
"""

import logging
import mmap
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")

DEVICE_PATH = "/dev/junior_spark"


@dataclass
class KernelPayload:
    ternary_data: bytes
    metadata: Dict[str, Any]
    coherence: float


class JuniorOSKernelBridge:
    def __init__(self):
        self.device_path = DEVICE_PATH
        self._mmap = None
        self._fd = None
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

            if self._mmap is not None:
                self._mmap.seek(0)
                self._mmap.write(byte_data[: len(self._mmap)])
                logging.debug("Wrote ternary manifold via mmap")
                return True

            with open(self.device_path, "wb") as f:
                f.write(byte_data)
            logging.debug("Wrote ternary manifold via regular file")
            return True

        except Exception as e:
            logging.warning(f"Kernel write failed: {e}")
            return False

    def _try_open_mmap(self):
        try:
            self._fd = os.open(self.device_path, os.O_RDWR)
            self._mmap = mmap.mmap(self._fd, 1048576, access=mmap.ACCESS_WRITE)
        except Exception as e:
            logging.debug(f"mmap not available, falling back to regular writes: {e}")
            self._mmap = None

    def close(self):
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._fd:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        self.close()
