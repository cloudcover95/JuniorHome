# path: src/juniorhome/application.py
#!/usr/bin/env python3
"""
Application

Main application bootstrap class for JuniorHome.
Initializes all core production components (Config, Logging, HealthCheck,
Metrics, DataLake, Docker, etc.) and provides a unified entry point.
"""

import logging
from typing import Optional

from .config_manager import ConfigManager
from .production_setup import ProductionSetup
from .health_check import HealthCheck
from .metrics_collector import MetricsCollector
from .datalake_manager import DataLakeManager
from .docker_manager import DockerManager
from .orchestrator import JuniorHomeOrchestrator

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class Application:
    """
    Main application class that bootstraps the full JuniorHome stack.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.config = ConfigManager(config_file)
        self.production = ProductionSetup(
            app_name=self.config.get("app_name", "JuniorHome"),
            log_level=self.config.get("log_level", "INFO"),
        )
        self.health = HealthCheck()
        self.metrics = MetricsCollector()
        self.datalake = DataLakeManager(
            base_path=self.config.get("data_dir", "data"),
            default_backend="parquet",
        )
        self.docker = DockerManager(
            project_name=self.config.get("app_name", "juniorhome").lower(),
        )
        self.orchestrator = JuniorHomeOrchestrator(config_path=config_file)

        self._register_default_health_checks()
        logging.info("Application bootstrap complete")

    def _register_default_health_checks(self):
        # Example health checks
        self.health.register_check("config_loaded", lambda: bool(self.config.to_dict()))
        self.health.register_check(
            "orchestrator_ready",
            lambda: self.orchestrator is not None,
        )

    def start(self):
        logging.info("Starting JuniorHome Application...")
        # Future: start scheduler, monitoring, etc.

    def shutdown(self):
        logging.info("Shutting down JuniorHome Application...")
        self.production.shutdown()

    def get_status(self):
        return {
            "config": self.config.to_dict(),
            "health": self.health.run_all_checks(),
            "metrics": self.metrics.get_all_metrics(),
            "orchestrator": self.orchestrator.status(),
        }
