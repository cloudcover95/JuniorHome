# path: src/juniorhome/orchestrator.py
#!/usr/bin/env python3
"""
JuniorHome Core Orchestrator

The central brain of the sovereign edge stack.
Coordinates DataLake, Reporter, plugins, swarm, BitNet-mlx, and memory.
"""

import logging
from typing import Any, Dict, Optional

from .config import JuniorHomeConfig
from .datalake import DataLake
from .plugin_loader import PluginLoader
from .reporter import Reporter

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class JuniorHomeOrchestrator:
    """
    Production-grade central orchestrator for JuniorCloud LLC stack.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = JuniorHomeConfig.from_file(config_path) if config_path else JuniorHomeConfig()

        self.datalake = DataLake(self.config.workspace_root)
        self.plugin_loader = PluginLoader()
        self.reporter: Optional[Reporter] = None

        # Placeholders for external components
        self.swarm = None
        self.bitnet_bridge = None
        self.memory_backend = None

        logging.info("JuniorHomeOrchestrator initialized")

    def register_swarm(self, swarm: Any):
        self.swarm = swarm
        logging.info("Swarm registered with orchestrator")

    def register_bitnet(self, bitnet_bridge: Any):
        self.bitnet_bridge = bitnet_bridge
        logging.info("BitNet-mlx bridge registered")

    def register_memory(self, memory_backend: Any):
        self.memory_backend = memory_backend
        logging.info("Memory backend registered")

    def initialize_reporter(self):
        self.reporter = Reporter(
            datalake=self.datalake,
            memory_backend=self.memory_backend,
            swarm=self.swarm,
            bitnet_bridge=self.bitnet_bridge,
        )
        logging.info("Reporter initialized")

    def generate_intelligent_report(self, topic: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.reporter:
            self.initialize_reporter()
        return self.reporter.generate_report(topic, context)

    def load_plugin(self, module_path: str, class_name: Optional[str] = None):
        return self.plugin_loader.load_plugin(module_path, class_name)

    def status(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "plugins_loaded": self.plugin_loader.list_plugins(),
            "has_swarm": self.swarm is not None,
            "has_bitnet": self.bitnet_bridge is not None,
            "has_memory": self.memory_backend is not None,
            "has_reporter": self.reporter is not None,
        }
