# path: src/juniorhome/junioros/kernel_bridge.py
#!/usr/bin/env python3
"""
JuniorOS Kernel Bridge (User-Space)

Production-grade user-space interface to /dev/junior_spark.
Supports both regular writes and zero-copy mmap when available.

Designed to be called by SovereignEdgeOrchestrator and TriStateExecutionEngine.
"""

import logging
import mmap
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
        else:
            logging.info("Running in user-space mode (no /dev/junior_spark)")

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
        """
        Write ternary manifold + metadata to the kernel.
        Uses mmap when possible for zero-copy behavior.
        """
        if not self._available:
            logging.debug("Kernel device not available. Skipping kernel write.")
            return False

        try:
            import numpy as np

            # Convert to compact int8 bytes
            if hasattr(ternary_tensor, "numpy"):
                arr = ternary_tensor.numpy().astype(np.int8)
            else:
                arr = np.asarray(ternary_tensor, dtype=np.int8)

            byte_data = arr.tobytes()

            payload = KernelPayload(
                ternary_data=byte_data,
                metadata=metadata or {},
                coherence=coherence,
            )

            # Try mmap first for zero-copy
            if self._mmap is None:
                self._open_mmap()

            if self._mmap is not None:
                # Simple ring-buffer style write (future: proper circular buffer)
                self._mmap.seek(0)
                self._mmap.write(byte_data[: len(self._mmap)])
                return True

            # Fallback to regular write
            with open(self.device_path, "wb") as f:
                f.write(byte_data)
            return True

        except Exception as e:
            logging.warning(f"Failed to write to kernel device: {e}")
            return False

    def _open_mmap(self):
        try:
            self._fd = os.open(self.device_path, os.O_RDWR)
            self._mmap = mmap.mmap(self._fd, 0, access=mmap.ACCESS_WRITE)
            logging.debug("Opened mmap to /dev/junior_spark")
        except Exception as e:
            logging.debug(f"mmap not available, using regular writes: {e}")
            self._mmap = None

    def close(self):
        if self._mmap:
            self._mmap.close()
        if self._fd:
            os.close(self._fd)
        logging.debug("Kernel bridge closed")

    def __del__(self):
        self.close()
