# path: src/juniorhome/orchestrator.py
#!/usr/bin/env python3
"""
JuniorHome Core Orchestrator (Updated)

Includes optional TernaryIntegration for BitNet-mlx ternary analysis.
"""

import logging
from typing import Any, Dict, Optional

from .config import JuniorHomeConfig
from .datalake import DataLake
from .plugin_loader import PluginLoader
from .reporter import Reporter
from .ternary_integration import TernaryIntegration

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class JuniorHomeOrchestrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = JuniorHomeConfig.from_file(config_path) if config_path else JuniorHomeConfig()

        self.datalake = DataLake(self.config.workspace_root)
        self.plugin_loader = PluginLoader()
        self.reporter: Optional[Reporter] = None
        self.ternary: Optional[TernaryIntegration] = None

        # Try to initialize ternary integration
        try:
            self.ternary = TernaryIntegration(
                output_dim=128,
                store_path=str(self.config.workspace_root) + "/ternary_signatures"
            )
            if self.ternary.is_available():
                logging.info("Ternary analysis enabled via BitNet-mlx")
        except Exception:
            logging.info("Ternary analysis not available")

        logging.info("JuniorHomeOrchestrator initialized")

    def register_swarm(self, swarm: Any):
        self.swarm = swarm
        logging.info("Swarm registered")

    def register_bitnet(self, bitnet_bridge: Any):
        self.bitnet_bridge = bitnet_bridge
        logging.info("BitNet-mlx bridge registered")

    def register_memory(self, memory_backend: Any):
        self.memory_backend = memory_backend
        logging.info("Memory backend registered")

    def initialize_reporter(self):
        self.reporter = Reporter(
            datalake=self.datalake,
            memory_backend=getattr(self, "memory_backend", None),
            swarm=getattr(self, "swarm", None),
            bitnet_bridge=getattr(self, "bitnet_bridge", None),
        )
        logging.info("Reporter initialized")

    def analyze_ternary(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.ternary and self.ternary.is_available():
            return self.ternary.analyze(data, metadata=metadata)
        return {"error": "Ternary analysis not available"}

    def generate_intelligent_report(self, topic: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.reporter:
            self.initialize_reporter()
        return self.reporter.generate_report(topic, context)

    def status(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "has_ternary": self.ternary.is_available() if self.ternary else False,
            "has_swarm": getattr(self, "swarm", None) is not None,
            "has_bitnet": getattr(self, "bitnet_bridge", None) is not None,
            "has_memory": getattr(self, "memory_backend", None) is not None,
        }
