# path: src/juniorhome/application.py
#!/usr/bin/env python3
"""
Application (with Second Brain + Service Locator)

Now registers the SecondBrain as a first-class service via the container.
"""

import logging
from typing import Optional

try:
    from .service_container import ServiceContainer
    HAS_CONTAINER = True
except ImportError:
    HAS_CONTAINER = False

# Core imports

from .config_manager import ConfigManager
from .production_setup import ProductionSetup
from .observability_manager import ObservabilityManager
from .security_middleware import SecurityMiddleware
from .datalake_manager import DataLakeManager
from .docker_manager import DockerManager
from .orchestrator import JuniorHomeOrchestrator
from .quantized_model_manager import QuantizedModelManager
from .knowledge_service import KnowledgeService
from .second_brain import SecondBrain

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Application:
    def __init__(self, config_file: Optional[str] = None):
        if not HAS_CONTAINER:
            logging.warning("ServiceContainer not available")

        self.container = ServiceContainer() if HAS_CONTAINER else None

        self.config = ConfigManager(config_file)
        self.production = ProductionSetup(
            app_name=self.config.get("app_name", "JuniorHome"),
            log_level=self.config.get("log_level", "INFO"),
        )

        if self.container:
            self.container.register(ConfigManager, self.config)
            self.container.register(ProductionSetup, self.production)

        self.observability = ObservabilityManager()
        if self.container:
            self.container.register(ObservabilityManager, self.observability)

        self.security = SecurityMiddleware(
            strict_mode=self.config.get("strict_security", True)
        )
        if self.container:
            self.container.register(SecurityMiddleware, self.security)

        self.datalake = DataLakeManager(base_path=self.config.get("data_dir", "data"))
        if self.container:
            self.container.register(DataLakeManager, self.datalake)

        # Second Brain (Obsidian + Data Lake)
        self.second_brain = SecondBrain(
            vault_path=self.config.get("obsidian_vault", "./obsidian"),
            data_dir=self.config.get("data_dir", "data"),
        )
        if self.container:
            self.container.register(SecondBrain, self.second_brain)

        self.docker = DockerManager(project_name=self.config.get("app_name", "juniorhome").lower())
        self.orchestrator = JuniorHomeOrchestrator(config_path=config_file)
        self.quantized_models = QuantizedModelManager()
        self.knowledge = KnowledgeService(
            vault_path=self.config.get("obsidian_vault", "./obsidian"),
            enable_scheduling=True,
        )

        logging.info("Application initialized with SecondBrain + ServiceContainer")

    def resolve(self, interface):
        if self.container:
            return self.container.resolve(interface)
        return None

    def start(self):
        logging.info("Starting JuniorHome Application...")

    def shutdown(self):
        logging.info("Shutting down JuniorHome Application...")
        self.production.shutdown()

    def get_full_status(self):
        return {
            "config": self.config.to_dict(),
            "observability": self.observability.get_full_status(),
            "security": self.security.get_security_status(),
            "orchestrator": self.orchestrator.status(),
            "quantized_models": self.quantized_models.list_loaded_models(),
            "second_brain_active": True,
        }
