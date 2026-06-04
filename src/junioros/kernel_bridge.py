# path: src/junioros/kernel_bridge.py

import asyncio
import logging
import mmap
import os
import struct
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

try:
    from ..bitnet.backends import router as backend_router
    HAS_BACKEND_ROUTER = True
except ImportError:
    HAS_BACKEND_ROUTER = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")

DEVICE_PATH = "/dev/junior_spark"
RING_SIZE = 2 * 1024 * 1024

MAGIC = 0x4A554E49
HEADER_FORMAT = "<I I f Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


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
        self._write_pos = 0
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
            return False

        try:
            arr = np.asarray(ternary_tensor, dtype=np.int8) if ternary_tensor is not None else b""
            ternary_bytes = arr.tobytes() if ternary_tensor is not None else b""

            payload = KernelPayload(
                ternary_data=ternary_bytes,
                metadata=metadata or {},
                coherence=coherence,
                timestamp=time.time(),
            )

            serialized = self._serialize_payload(payload)

            if self._mmap is None:
                self._try_open_mmap()

            if self._mmap is not None:
                self._write_to_ring(serialized)
            else:
                with open(self.device_path, "wb") as f:
                    f.write(serialized)

            return True

        except Exception:
            return False

    def _serialize_payload(self, payload: KernelPayload) -> bytes:
        meta_bytes = str(payload.metadata).encode("utf-8")
        meta_len = struct.pack("<I", len(meta_bytes))

        total_len = HEADER_SIZE + 4 + len(meta_bytes) + len(payload.ternary_data)
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC,
            total_len,
            payload.coherence,
            int(payload.timestamp * 1_000_000),
        )

        return header + meta_len + meta_bytes + payload.ternary_data

    def _write_to_ring(self, data: bytes):
        if self._mmap is None:
            return

        data_len = len(data)
        if data_len > RING_SIZE:
            return

        if self._write_pos + data_len > RING_SIZE:
            self._write_pos = 0

        self._mmap.seek(self._write_pos)
        self._mmap.write(data)
        self._write_pos = (self._write_pos + data_len) % RING_SIZE

    def _try_open_mmap(self):
        try:
            self._fd = os.open(self.device_path, os.O_RDWR)
            self._mmap = mmap.mmap(self._fd, RING_SIZE, access=mmap.ACCESS_WRITE)
        except Exception:
            self._mmap = None

    async def async_write_ternary_manifold(
        self,
        ternary_tensor: Any,
        metadata: Optional[Dict[str, Any]] = None,
        coherence: float = 0.0,
    ) -> bool:
        return await asyncio.to_thread(
            self.write_ternary_manifold,
            ternary_tensor,
            metadata,
            coherence,
        )

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
