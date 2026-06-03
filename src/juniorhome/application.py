# path: src/juniorhome/application.py
#!/usr/bin/env python3
"""
Application (Architecture v3 - with Interfaces)

Now uses interfaces + ServiceContainer for strong dependency inversion.
"""

import logging
from typing import Optional

try:
    from .service_container import ServiceContainer
    HAS_CONTAINER = True
except ImportError:
    HAS_CONTAINER = False

# Interfaces
try:
    from .interfaces import IDataLake, ISecondBrain, IOrchestrator, IKnowledgeService
except ImportError:
    IDataLake = ISecondBrain = IOrchestrator = IKnowledgeService = object

# Implementations

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

        self.datalake: IDataLake = DataLakeManager(base_path=self.config.get("data_dir", "data"))
        if self.container:
            self.container.register(IDataLake, self.datalake)

        self.second_brain: ISecondBrain = SecondBrain(
            vault_path=self.config.get("obsidian_vault", "./obsidian"),
            data_dir=self.config.get("data_dir", "data"),
        )
        if self.container:
            self.container.register(ISecondBrain, self.second_brain)

        self.docker = DockerManager(project_name=self.config.get("app_name", "juniorhome").lower())
        self.orchestrator: IOrchestrator = JuniorHomeOrchestrator(config_path=config_file)
        if self.container:
            self.container.register(IOrchestrator, self.orchestrator)

        self.quantized_models = QuantizedModelManager()
        self.knowledge: IKnowledgeService = KnowledgeService(
            vault_path=self.config.get("obsidian_vault", "./obsidian"),
            enable_scheduling=True,
        )
        if self.container:
            self.container.register(IKnowledgeService, self.knowledge)

        logging.info("Application v3 initialized with interfaces + ServiceContainer")

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
