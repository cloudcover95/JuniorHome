# path: src/juniorhome/orchestrator.py
#!/usr/bin/env python3
"""
JuniorHome Core Orchestrator (Final Integration)

Includes ServiceRegistry, ModuleLoader, PluginSystem,
Ollama, BitNet-mlx ternary pipeline, and SmartLLMRouter.
"""

import logging
from typing import Any, Dict, Optional

from .config import JuniorHomeConfig
from .datalake import DataLake
from .module_loader import ModuleLoader
from .plugin_loader import PluginLoader
from .plugin_system import PluginSystem
from .reporter import Reporter
from .service_registry import ServiceRegistry
from .smart_llm_router import SmartLLMRouter
from .ternary_integration import TernaryIntegration

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class JuniorHomeOrchestrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = JuniorHomeConfig.from_file(config_path) if config_path else JuniorHomeConfig()

        self.datalake = DataLake(self.config.workspace_root)
        self.module_loader = ModuleLoader()
        self.service_registry = ServiceRegistry()
        self.plugin_system = PluginSystem(
            module_loader=self.module_loader,
            service_registry=self.service_registry
        )

        self.reporter: Optional[Reporter] = None
        self.ternary = TernaryIntegration()
        self.llm_router = SmartLLMRouter()

        logging.info("JuniorHomeOrchestrator initialized with full stack")

    def analyze_ternary(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.ternary and self.ternary.is_available():
            return self.ternary.analyze(data, metadata=metadata)
        return {"error": "Ternary analysis not available"}

    def route_llm(
        self,
        prompt: str,
        prefer_bitnet: bool = False,
        model: str = "llama3.2",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.llm_router.route(
            prompt=prompt,
            prefer_bitnet=prefer_bitnet,
            model=model,
            metadata=metadata,
        )

    def generate_intelligent_report(self, topic: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.reporter:
            self.reporter = Reporter(
                datalake=self.datalake,
                memory_backend=None,
                swarm=None,
                bitnet_bridge=None,
            )
        return self.reporter.generate_report(topic, context)

    def status(self) -> Dict[str, Any]:
        return {
            "ollama_available": self.llm_router.ollama_available,
            "bitnet_available": self.llm_router.bitnet_available,
            "ternary_available": self.ternary.is_available() if self.ternary else False,
            "plugins_loaded": self.plugin_system.list_plugins(),
            "services_registered": self.service_registry.list_services(),
        }
