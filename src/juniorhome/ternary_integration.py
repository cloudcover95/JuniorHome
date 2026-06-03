# path: src/juniorhome/ternary_integration.py
#!/usr/bin/env python3
"""
Ternary Integration

Bridge between JuniorHome and the BitNet-mlx ternary analysis pipeline.
Allows the orchestrator to easily run ternary projection + TDA
on telemetry and spatial data.
"""

import logging
from typing import Any, Dict, Optional

try:
    from bitnet_mlx.inference.ternary_pipeline import TernaryPipeline
    HAS_BITNET = True
except ImportError:
    HAS_BITNET = False
    TernaryPipeline = None

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class TernaryIntegration:
    """
    Provides access to the BitNet-mlx ternary pipeline from JuniorHome.
    """

    def __init__(self, output_dim: int = 128, store_path: Optional[str] = None):
        if not HAS_BITNET:
            logging.warning("bitnet_mlx not available. Ternary features disabled.")
            self.pipeline = None
        else:
            self.pipeline = TernaryPipeline(output_dim=output_dim, store_path=store_path)
            logging.info("TernaryIntegration initialized")

    def analyze(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.pipeline is None:
            return {"error": "BitNet-mlx not available"}

        return self.pipeline.run(data, metadata=metadata)

    def analyze_batch(self, batch: list, metadata: Optional[Dict[str, Any]] = None) -> list:
        if self.pipeline is None:
            return [{"error": "BitNet-mlx not available"} for _ in batch]

        return self.pipeline.run_batch(batch, metadata=metadata)

    def is_available(self) -> bool:
        return self.pipeline is not None
