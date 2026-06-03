# path: src/juniorhome/application.py
#!/usr/bin/env python3
"""
Application (Strengthened)

The central architectural hub of JuniorHome.
Bootstraps and wires all major subsystems (Config, Observability,
Security, Data, Agents, Quantization, Knowledge, etc.).
"""

import logging
from typing import Optional

from .config_manager import ConfigManager
from .production_setup import ProductionSetup
from .observability_manager import ObservabilityManager
from .security_middleware import SecurityMiddleware
from .datalake_manager import DataLakeManager
from .docker_manager import DockerManager
from .orchestrator import JuniorHomeOrchestrator
from .quantized_model_manager import QuantizedModelManager
from .knowledge_service import KnowledgeService

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Application:
    """
    Central architectural hub and bootstrap for the entire JuniorHome stack.
    """

    def __init__(self, config_file: Optional[str] = None):
        # Core infrastructure
        self.config = ConfigManager(config_file)
        self.production = ProductionSetup(
            app_name=self.config.get("app_name", "JuniorHome"),
            log_level=self.config.get("log_level", "INFO"),
        )

        # Observability
        self.observability = ObservabilityManager()

        # Security
        self.security = SecurityMiddleware(
            strict_mode=self.config.get("strict_security", True)
        )

        # Data
        self.datalake = DataLakeManager(
            base_path=self.config.get("data_dir", "data")
        )

        # Deployment
        self.docker = DockerManager(
            project_name=self.config.get("app_name", "juniorhome").lower()
        )

        # Core Orchestrator
        self.orchestrator = JuniorHomeOrchestrator(config_path=config_file)

        # Quantized Models (BitNet-mlx)
        self.quantized_models = QuantizedModelManager()

        # Knowledge Processing
        self.knowledge = KnowledgeService(
            vault_path=self.config.get("obsidian_vault", "./obsidian"),
            enable_scheduling=True,
        )

        logging.info("Application (central architecture hub) initialized")

    def start(self):
        logging.info("Starting JuniorHome Application...")
        # Future: auto-start scheduler, websocket, etc.

    def shutdown(self):
        logging.info("Shutting down JuniorHome Application...")
        self.production.shutdown()
        if hasattr(self.knowledge, "shutdown"):
            self.knowledge.shutdown()

    def get_full_status(self):
        return {
            "config": self.config.to_dict(),
            "observability": self.observability.get_full_status(),
            "security": self.security.get_security_status(),
            "orchestrator": self.orchestrator.status(),
            "quantized_models": self.quantized_models.list_loaded_models(),
            "knowledge_service_active": True,
        }
