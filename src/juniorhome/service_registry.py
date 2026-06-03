# path: src/juniorhome/service_registry.py
#!/usr/bin/env python3
"""
Service Registry

Core service registry for JuniorHome.
Provides dynamic registration, discovery, and lifecycle management
of services and agents. Foundational primitive for future BitNet OS.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class ServiceRegistry:
    """
    Lightweight service registry with support for dependency injection
    and lifecycle hooks. Designed for dynamic, local-first runtimes.
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        logging.info("ServiceRegistry initialized")

    def register(self, name: str, instance: Any, dependencies: Optional[List[str]] = None):
        self._services[name] = instance
        if dependencies:
            self._dependencies[name] = dependencies
        logging.info(f"Service registered: {name}")

    def register_factory(self, name: str, factory: Callable[[], Any], dependencies: Optional[List[str]] = None):
        self._factories[name] = factory
        if dependencies:
            self._dependencies[name] = dependencies
        logging.info(f"Service factory registered: {name}")

    def get(self, name: str) -> Optional[Any]:
        if name in self._services:
            return self._services[name]

        if name in self._factories:
            instance = self._factories[name]()
            self._services[name] = instance
            return instance

        logging.warning(f"Service not found: {name}")
        return None

    def list_services(self) -> List[str]:
        return list(set(list(self._services.keys()) + list(self._factories.keys())))

    def has(self, name: str) -> bool:
        return name in self._services or name in self._factories

    def remove(self, name: str) -> bool:
        if name in self._services:
            del self._services[name]
            logging.info(f"Service removed: {name}")
            return True
        if name in self._factories:
            del self._factories[name]
            logging.info(f"Service factory removed: {name}")
            return True
        return False
